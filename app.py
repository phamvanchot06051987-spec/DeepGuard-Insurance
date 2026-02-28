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
# 2. 完整 UI 界面 (白标化 + B端展业功能)
# ==========================================
def main():
    st.set_page_config(page_title="智能保单管家", layout="centered", page_icon="🛡️", initial_sidebar_state="collapsed")
    
    # 🔴 核心：商业级“掩耳盗铃” CSS，彻底隐藏所有 Streamlit 和 GitHub 标志
    st.markdown("""
        <style>
        /* 隐藏顶部红线和页眉 */
        [data-testid="stHeader"] {display: none;}
        .block-container {padding-top: 1rem; padding-bottom: 1rem;}
        
        /* 隐藏右上角菜单和 Deploy 按钮 */
        #MainMenu {visibility: hidden;}
        .stDeployButton {display: none;}
        
        /* 隐藏底部 Made with Streamlit */
        footer {visibility: hidden;}
        
        /* 隐藏右下角 GitHub 链接 (核心白标化) */
        .viewerBadge_container__1QSob {display: none !important;}
        [data-testid="stStatusWidget"] {display: none !important;}
        
        /* 美化看板指标卡片 */
        [data-testid="stMetricValue"] {font-size: 1.8rem; color: #1E88E5;}
        </style>
    """, unsafe_allow_html=True)

    st.title("🛡️ 智能家庭保险中台")
    
    if 'api_key' not in st.session_state:
        st.session_state.api_key = ""
        
    ds_key = st.session_state.api_key
    sys = UltimateInsuranceSystem(ds_key)
    
    if not ds_key:
        st.warning("👈 代理人请先前往【⚙️ 设置】配置您的 API Key。")

    t1, t2, t3, t4, t5 = st.tabs(["📸 拍照解析", "👨‍👩‍👧‍👦 家族看板", "🚗 车险理赔", "💬 条款顾问", "⚙️ 设置"])

    # ---------------- 模块1：拍照与解析 ----------------
    with t1:
        st.subheader("一键提取旧保单")
        img_file = st.file_uploader("拍摄或上传保单照片", type=['jpg', 'png', 'jpeg'])
        if img_file:
            with st.spinner("图像增强中..."):
                st.image(sys.process_image(img_file.getvalue()), caption="增强效果图", use_container_width=True)
        
        raw_text = st.text_area("粘贴 OCR 文本内容：", height=100)
        if st.button("🚀 AI 结构化解析", use_container_width=True):
            if raw_text and ds_key:
                with st.spinner("精算引擎解析中..."):
                    res = sys.analyze_policy(raw_text)
                    if res:
                        st.session_state.temp_res = res
                        st.success("解析成功！")
            elif not ds_key: st.error("请配置 API Key")

        if 'temp_res' in st.session_state:
            edit_df = st.data_editor(pd.DataFrame([st.session_state.temp_res]), use_container_width=True)
            if st.button("✅ 存入客户家庭库", use_container_width=True):
                r = edit_df.iloc[0].to_dict()
                sys.conn.execute("INSERT INTO family_atlas (holder, insured, relation, product, premium, coverage, category, raw_data) VALUES (?,?,?,?,?,?,?,?)",
                                 (r['holder'], r['insured'], r['relation'], r['product'], r['premium'], r['coverage'], r.get('category','综合'), json.dumps(r)))
                sys.conn.commit()
                st.success("已归档！请前往【家族看板】查看或生成报告。")
                del st.session_state.temp_res

    # ---------------- 模块2：家族看板 & 展业报告生成 ----------------
    with t2:
        df = pd.read_sql("SELECT * FROM family_atlas", sys.conn)
        if not df.empty:
            c1, c2 = st.columns(2)
            c1.metric("家庭年缴保费", f"¥{df['premium'].sum():,.2f}")
            c2.metric("覆盖成员数", f"{len(df['insured'].unique())} 人")
            st.bar_chart(df, x="insured", y="premium", color="category")

            st.divider()
            st.subheader("💼 B端专属：一键生成展业体检报告")
            st.caption("利用大模型分析当前数据，自动生成用于促单的专业报告。")
            
            col_a, col_b = st.columns(2)
            agent_name = col_a.text_input("您的姓名 (理财师)", placeholder="例如：李总监")
            client_name = col_b.text_input("客户称呼", placeholder="例如：张先生")
            
            if st.button("✨ 生成专属 PDF/Markdown 报告", use_container_width=True, type="primary"):
                if agent_name and client_name and ds_key:
                    with st.spinner("AI 正在撰写专业理财报告..."):
                        report_content = sys.generate_professional_report(df.to_json(orient="records", force_ascii=False), agent_name, client_name)
                        st.session_state.generated_report = report_content
                else:
                    st.error("请填写理财师姓名、客户称呼，并确保配置了 API Key。")
                    
            # 显示报告并提供【打印为PDF】按钮
            if 'generated_report' in st.session_state:
                st.markdown("---")
                st.markdown("<div style='background-color:#ffffff; padding:20px; border-radius:10px; border:1px solid #e0e0e0; color:#333;'>", unsafe_allow_html=True)
                st.markdown(st.session_state.generated_report)
                st.markdown("</div>", unsafe_allow_html=True)
                
                st.write("")
                # 利用 JavaScript 调用浏览器的打印功能，完美导出 PDF
                components.html(
                    """
                    <script>
                    function printReport() {
                        window.parent.print();
                    }
                    </script>
                    <button onclick="printReport()" style="width: 100%; padding: 10px; background-color: #4CAF50; color: white; border: none; border-radius: 5px; font-size: 16px; cursor: pointer;">
                        🖨️ 保存为 PDF (调用浏览器打印)
                    </button>
                    <p style="text-align:center; font-size:12px; color:gray; font-family:sans-serif;">提示：点击后在弹出的窗口中选择“另存为 PDF”即可发送给客户。</p>
                    """, height=100
                )
        else:
            st.info("暂无数据，请先在【拍照解析】中录入客户保单。")

    # ---------------- 模块3：车险理赔预判 ----------------
    with t3:
        st.subheader("🚗 车险事故向导")
        acc_text = st.text_area("描述事故经过...", height=150)
        if st.button("⚖️ 生成建议", use_container_width=True):
            if acc_text and ds_key:
                st.markdown(sys.predict_auto_claim(acc_text))

    # ---------------- 模块4：条款智能问答 ----------------
    with t4:
        st.subheader("💬 长文本条款顾问")
        pdf_file = st.file_uploader("上传保险合同 (PDF)", type=['pdf'])
        question = st.text_input("向条款提问")
        if st.button("💡 查阅并解答", use_container_width=True):
            if pdf_file and question and ds_key:
                reader = PyPDF2.PdfReader(BytesIO(pdf_file.getvalue()))
                full_text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])
                if full_text:
                    st.write(sys.qa_policy_terms(full_text, question))
                else:
                    st.error("无法提取文本，可能是纯图片PDF。")

    # ---------------- 模块5：系统设置 ----------------
    with t5:
        st.subheader("⚙️ 引擎配置")
        new_key = st.text_input("DeepSeek API Key", type="password", value=st.session_state.api_key)
        if st.button("💾 保存生效", use_container_width=True):
            st.session_state.api_key = new_key
            st.success("配置已保存！")
            st.rerun()

if __name__ == "__main__":
    main()
