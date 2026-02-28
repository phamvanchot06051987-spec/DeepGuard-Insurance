import streamlit as st
import pandas as pd
import sqlite3
import json
import re
import cv2
import numpy as np
import PyPDF2  
from io import BytesIO
import streamlit.components.v1 as components

from langchain_openai import ChatOpenAI

# ==========================================
# 1. 全功能核心引擎：DeepSeek 驱动
# ==========================================
class UltimateInsuranceSystem:
    def __init__(self, api_key="", model="deepseek-chat"):
        self.api_key = api_key
        self.llm = None
        if api_key:
            self.llm = ChatOpenAI(
                model=model,
                api_key=api_key,
                base_url="https://api.deepseek.com",
                temperature=0.1,  
                max_tokens=4000
            )
        self.conn = sqlite3.connect('insurance_ultimate.db', check_same_thread=False)
        self._init_db()

    def _init_db(self):
        self.conn.execute('''CREATE TABLE IF NOT EXISTS family_atlas 
            (id INTEGER PRIMARY KEY AUTOINCREMENT, 
             holder TEXT, insured TEXT, relation TEXT, 
             product TEXT, premium REAL, coverage REAL, 
             category TEXT, raw_data TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
        self.conn.commit()

    def process_image(self, image_bytes):
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    def analyze_policy(self, raw_text):
        if not self.llm: return None
        text = re.sub(r'\s+', ' ', raw_text.strip()) if hasattr(raw_text, 'strip') else raw_text
        system_prompt = """你是一个专业的保险精算助手。请从文本提取关键信息并返回JSON。
        字段要求：holder(投保人), insured(被保人), relation(关系:本人/配偶/子女/父母), product(产品名), premium(保费数字), coverage(保额数字), category(险种:重疾/医疗/意外/寿险/车险)。
        仅返回 JSON 代码块。"""
        try:
            res = self.llm.invoke([("system", system_prompt), ("user", f"提取内容：\n{text}")])
            match = re.search(r'\{.*\}', res.content, re.DOTALL)
            return json.loads(match.group()) if match else json.loads(res.content)
        except Exception as e:
            st.error(f"解析异常: {e}")
            return None

    def predict_auto_claim(self, accident_desc):
        if not self.llm: return "请先配置 API Key"
        prompt = f"""你是一位资深车险理赔专家。用户描述了交通事故："{accident_desc}"
        请分析：1. 责任判定； 2. 动用险种； 3. 理赔避坑建议。Markdown格式回复。"""
        return self.llm.invoke(prompt).content

    def qa_policy_terms(self, pdf_text, question):
        if not self.llm: return "请先配置 API Key"
        prompt = f"""你是一位保险律师。根据条款原文节选回答客户问题。
        原文：{pdf_text[:15000]}...
        问题："{question}"。
        必须依据条款原文回答，指出理赔条件或免责情况。"""
        return self.llm.invoke(prompt).content
        
    def generate_professional_report(self, family_data, agent_name, client_name):
        """生成 B 端展业专属 PDF/Markdown 报告"""
        if not self.llm: return "请先配置 API Key"
        prompt = f"""
        你是一名顶级的百万圆桌(MDRT)保险理财师 {agent_name}。
        请根据以下家庭现有的保单数据，为尊贵的客户 {client_name} 撰写一份专业的《家庭保障体检与规划报告》。
        
        家庭已存保单数据：
        {family_data}
        
        要求撰写以下板块（使用严谨、客观、有温度的专家口吻）：
        1. 🌟 卷首语：感谢信任。
        2. 📊 现状诊断：总结目前家庭交了多少钱，保了哪些人，主要集中在什么险种。
        3. 🚨 风险缺口暴露：一针见血地指出家庭目前的巨大风险（例如：家庭支柱缺乏重疾险、某成员缺乏医疗险、保额是否足以覆盖大病等）。这是促单的核心！
        4. 💡 专属优化方案：给出具体的加保方向和思路（不提具体产品名）。
        5. 🤝 结语：署名 {agent_name}。
        
        请使用精美的 Markdown 格式排版。
        """
        return self.llm.invoke(prompt).content

# ==========================================
# 2. UI 界面：白标化样式 + 档案管理逻辑
# ==========================================
def main():
    st.set_page_config(page_title="智能保单管家", layout="centered", page_icon="🛡️", initial_sidebar_state="collapsed")
    
    # 🔴 白标化 CSS：隐藏所有官方标志
    st.markdown("""
        <style>
        [data-testid="stHeader"] {display: none;}
        .block-container {padding-top: 1rem;}
        #MainMenu {visibility: hidden;}
        .stDeployButton {display: none;}
        footer {visibility: hidden;}
        .viewerBadge_container__1QSob {display: none !important;}
        [data-testid="stStatusWidget"] {display: none !important;}
        .stMetric { background-color: #ffffff; padding: 20px; border-radius: 12px; border: 1px solid #f0f2f6; box-shadow: 0 4px 6px rgba(0,0,0,0.03); }
        div[data-testid="stMetricValue"] { color: #1e88e5; font-weight: bold; }
        </style>
    """, unsafe_allow_html=True)

    # 初始化状态
    if 'api_key' not in st.session_state: st.session_state.api_key = ""
    if 'current_family' not in st.session_state: st.session_state.current_family = "未命名档案"
        
    ds_key = st.session_state.api_key
    sys = UltimateInsuranceSystem(ds_key)

    # --- 顶层客户档案管理 ---
    st.markdown("### 🗃️ 客户档案管理中台")
    all_families = sys.get_all_families()
    
    col_f1, col_f2 = st.columns([2, 1])
    
    # 档案切换逻辑
    if not all_families:
        selected_family = col_f1.selectbox("当前选中的客户：", ["暂无客户档案，请先创建"])
    else:
        if st.session_state.current_family not in all_families:
            st.session_state.current_family = all_families[0]
        selected_family = col_f1.selectbox("选择或切换客户档案：", all_families, index=all_families.index(st.session_state.current_family))
        st.session_state.current_family = selected_family

    # 新增档案逻辑
    new_family_input = col_f2.text_input("➕ 新增客户档案", placeholder="如：张三家庭")
    if col_f2.button("创建新档案", use_container_width=True):
        if new_family_input:
            if new_family_input not in all_families:
                st.session_state.current_family = new_family_input
                st.success(f"档案【{new_family_input}】创建成功！")
                st.rerun()
            else:
                st.error("该档案名称已存在")

    st.divider()

    if not ds_key:
        st.warning("⚠️ 欢迎使用！请先前往【⚙️ 设置】填入 API Key 激活核心 AI 引擎。")

    t1, t2, t3, t4, t5 = st.tabs(["📸 保单导入", "👨‍👩‍👧‍👦 档案看板", "🚗 车险理赔", "💬 条款顾问", "⚙️ 设置"])

    # --- 1. 拍照与解析 (关联当前家庭) ---
    with t1:
        st.subheader(f"📄 录入保单：归属于 【{st.session_state.current_family}】")
        img_file = st.file_uploader("上传照片/截图进行增强", type=['jpg', 'png', 'jpeg'])
        if img_file:
            st.image(sys.process_image(img_file.getvalue()), caption="增强处理预览", use_container_width=True)
        
        raw_text = st.text_area("请粘贴 OCR 识别文本：", height=120, placeholder="此处粘贴保单文字内容...")
        if st.button("🚀 启动 AI 结构化解析", use_container_width=True, type="primary"):
            if raw_text and ds_key:
                with st.spinner("DeepSeek 正在解析..."):
                    res = sys.analyze_policy(raw_text)
                    if res: st.session_state.temp_res = res
            elif not ds_key: st.error("请先配置 API Key")

        if 'temp_res' in st.session_state:
            st.info(f"💡 提取结果将保存至客户：**{st.session_state.current_family}**")
            edit_df = st.data_editor(pd.DataFrame([st.session_state.temp_res]), use_container_width=True)
            if st.button("✅ 确认入库并归档", use_container_width=True):
                r = edit_df.iloc[0].to_dict()
                sys.conn.execute(
                    "INSERT INTO family_atlas (family_name, holder, insured, relation, product, premium, coverage, category, raw_data) VALUES (?,?,?,?,?,?,?,?,?)",
                    (st.session_state.current_family, r['holder'], r['insured'], r['relation'], r['product'], r['premium'], r['coverage'], r.get('category','综合'), json.dumps(r))
                )
                sys.conn.commit()
                st.success(f"归档成功！数据已存入【{st.session_state.current_family}】档案库。")
                del st.session_state.temp_res

    # --- 2. 客户档案看板 (支持数据筛选与全局概览) ---
    with t2:
        # 1. 当前选中档案的数据
        df_current = pd.read_sql(f"SELECT * FROM family_atlas WHERE family_name = '{st.session_state.current_family}'", sys.conn)
        
        # 2. 所有客户的全局简报
        df_all = pd.read_sql("SELECT family_name, SUM(premium) as total_premium, COUNT(*) as policy_count FROM family_atlas GROUP BY family_name", sys.conn)

        st.subheader(f"📊 【{st.session_state.current_family}】 专属全景看板")
        if not df_current.empty:
            col_m1, col_m2 = st.columns(2)
            col_m1.metric("家庭年缴总保费", f"¥{df_current['premium'].sum():,.2f}")
            col_m2.metric("已录入保单数", f"{len(df_current)} 份")
            
            st.bar_chart(df_current, x="insured", y="premium", color="category")
            
            # 报告生成区
            st.divider()
            st.subheader("💼 专业展业报告生成")
            agent_name = st.text_input("您的姓名/头衔", value="资深保险规划师")
            if st.button("✨ 生成 PDF 诊断报告内容", use_container_width=True, type="primary"):
                if ds_key:
                    with st.spinner("正在分析保障漏洞并撰写报告..."):
                        report = sys.generate_professional_report(df_current.to_json(orient="records", force_ascii=False), agent_name, st.session_state.current_family)
                        st.session_state.final_report = report
                else: st.error("请先配置 API Key")

            if 'final_report' in st.session_state:
                st.markdown(f"<div style='background-color:white; color:#333; padding:30px; border-radius:12px; border:1px solid #ddd; margin-top:20px;'>{st.session_state.final_report}</div>", unsafe_allow_html=True)
                components.html("""<script>function p(){ window.parent.print(); }</script><button onclick="p()" style="width:100%; padding:12px; background:#4CAF50; color:white; border:none; border-radius:8px; cursor:pointer; font-size:16px; font-weight:bold; margin-top:15px;">🖨️ 打印/导出 PDF 报告</button>""", height=80)
        else:
            st.info(f"档案【{st.session_state.current_family}】下暂无数据。")

        # 3. 全局客户大盘列表
        st.divider()
        st.subheader("🌍 代理人全局视角：所有客户概览")
        if not df_all.empty:
            st.dataframe(df_all.rename(columns={'family_name':'客户档案名', 'total_premium':'总保费', 'policy_count':'保单数量'}), use_container_width=True, hide_index=True)
        else:
            st.caption("暂无任何客户档案记录。")

    # --- 3. 车险理赔 (独立功能) ---
    with t3:
        st.subheader("🚗 车险事故理赔预判")
        acc_text = st.text_area("请详细描述事故经过：", height=150, placeholder="如：我在直行时撞到了左转的车辆，我是全责吗？报保险划算吗？")
        if st.button("⚖️ 启动 AI 风险评估", use_container_width=True):
            if ds_key and acc_text:
                with st.spinner("AI 专家推演中..."):
                    st.markdown(sys.predict_auto_claim(acc_text))

    # --- 4. 条款问答 (独立功能) ---
    with t4:
        st.subheader("💬 长文本条款智能律师")
        pdf_file = st.file_uploader("上传保险合同全文 PDF", type=['pdf'])
        question = st.text_input("针对条款提问：", placeholder="如：等待期内查出囊肿赔吗？既往症如何定义？")
        if st.button("💡 立即穿透条款查询", use_container_width=True):
            if pdf_file and question and ds_key:
                with st.spinner("正在查阅数万字条款原文..."):
                    reader = PyPDF2.PdfReader(BytesIO(pdf_file.getvalue()))
                    full_text = "".join([p.extract_text() for p in reader.pages if p.extract_text()])
                    st.write(sys.qa_policy_terms(full_text, question))

    # --- 5. 设置 ---
    with t5:
        st.subheader("⚙️ 系统引擎设置")
        new_key = st.text_input("DeepSeek API Key (API 密钥)", type="password", value=st.session_state.api_key)
        if st.button("💾 保存并更新配置", use_container_width=True):
            st.session_state.api_key = new_key
            st.success("配置已保存！各项 AI 引擎已激活。")
            st.rerun()

if __name__ == "__main__":
    main()
