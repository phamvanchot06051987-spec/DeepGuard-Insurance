import streamlit as st
import pandas as pd
import sqlite3
import json
import re
import cv2
import numpy as np
import PyPDF2  # 用于处理上传的 PDF 条款
from io import BytesIO

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
                             (
                                 id
                                 INTEGER
                                 PRIMARY
                                 KEY
                                 AUTOINCREMENT,
                                 holder
                                 TEXT,
                                 insured
                                 TEXT,
                                 relation
                                 TEXT,
                                 product
                                 TEXT,
                                 premium
                                 REAL,
                                 coverage
                                 REAL,
                                 category
                                 TEXT,
                                 raw_data
                                 TEXT,
                                 created_at
                                 TIMESTAMP
                                 DEFAULT
                                 CURRENT_TIMESTAMP
                             )''')
        self.conn.commit()

    def process_image(self, image_bytes):
        """【模块1】图像视觉预处理 (OpenCV)"""
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 自适应二值化增强文字清晰度
        enhanced = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        return enhanced

    def analyze_policy(self, raw_text):
        """【模块2】保单结构化提取 (DeepSeek)"""
        if not self.llm: return None
        text = re.sub(r'\s+', ' ', text.strip()) if hasattr(raw_text, 'strip') else raw_text
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
        """【模块3】车险理赔智能预判"""
        if not self.llm: return "请先配置 API Key"
        prompt = f"""你是一位资深车险理赔专家。用户描述了以下交通事故：
        "{accident_desc}"
        请分析：1. 责任判定预估； 2. 应该动用哪些险种（交强险/三者险/车损险）； 3. 理赔避坑建议。
        请用清晰的Markdown分点回复。"""
        return self.llm.invoke(prompt).content

    def qa_policy_terms(self, pdf_text, question):
        """【模块4】条款文档智能问答 (长上下文替代复杂RAG)"""
        if not self.llm: return "请先配置 API Key"
        # DeepSeek 支持长上下文，直接将条款文本喂给大模型即可实现精准问答
        prompt = f"""你是一位保险律师。以下是保险合同条款的原文节选：
        {pdf_text[:15000]} ... (截取部分)

        请根据以上条款，回答客户的问题："{question}"。
        要求：必须依据条款原文回答，指出理赔条件或免责情况。"""
        return self.llm.invoke(prompt).content


# ==========================================
# 2. 完整 UI 界面 (移动端自适应)
# ==========================================
def main():
    # 因为已经移除了侧边栏，这里把 initial_sidebar_state 改回 "collapsed" 使界面更整洁
    st.set_page_config(page_title="DeepGuard 全能版", layout="centered", page_icon="🛡️",
                       initial_sidebar_state="collapsed")
    st.markdown("""<style>[data-testid="stHeader"] {display: none;} .block-container {padding-top: 1rem;}</style>""",
                unsafe_allow_html=True)

    st.title("🛡️ DeepGuard 智能保险管家")

    # 引入 session_state 缓存 API Key，防止页面刷新时丢失
    if 'api_key' not in st.session_state:
        st.session_state.api_key = ""

    ds_key = st.session_state.api_key
    sys = UltimateInsuranceSystem(ds_key)

    if not ds_key:
        st.warning("👈 请先前往【⚙️ 设置】标签页配置 API Key，以激活全量 AI 功能。")

    # 五大功能模块 Tab（新增了 设置 Tab）
    t1, t2, t3, t4, t5 = st.tabs(["📸 拍照解析", "👨‍👩‍👧‍👦 家族看板", "🚗 车险理赔", "💬 条款顾问", "⚙️ 设置"])

    # ---------------- 模块1：拍照与解析 ----------------
    with t1:
        st.subheader("1. 视觉预处理")
        img_file = st.file_uploader("拍摄或上传保单照片", type=['jpg', 'png', 'jpeg'])
        if img_file:
            with st.spinner("OpenCV 正在增强图像..."):
                proc_img = sys.process_image(img_file.getvalue())
                st.image(proc_img, caption="二值化增强效果 (可提高 OCR 识别率)", use_container_width=True)

        st.subheader("2. 智能提取入库")
        raw_text = st.text_area("输入或粘贴 OCR 文本内容：", height=120)
        if st.button("🚀 结构化解析", use_container_width=True):
            if raw_text and ds_key:
                with st.spinner("AI 精算引擎解析中..."):
                    res = sys.analyze_policy(raw_text)
                    if res:
                        st.session_state.temp_res = res
                        st.success("解析成功！")
            elif not ds_key:
                st.error("请配置 API Key")

        if 'temp_res' in st.session_state:
            edit_df = st.data_editor(pd.DataFrame([st.session_state.temp_res]), use_container_width=True)
            if st.button("✅ 确认归档至家族图谱", use_container_width=True):
                r = edit_df.iloc[0].to_dict()
                sys.conn.execute(
                    "INSERT INTO family_atlas (holder, insured, relation, product, premium, coverage, category, raw_data) VALUES (?,?,?,?,?,?,?,?)",
                    (r['holder'], r['insured'], r['relation'], r['product'], r['premium'], r['coverage'],
                     r.get('category', '综合'), json.dumps(r)))
                sys.conn.commit()
                st.balloons()
                st.success("已存入数据库！请查看【家族看板】")
                del st.session_state.temp_res

    # ---------------- 模块2：家族看板 ----------------
    with t2:
        df = pd.read_sql("SELECT * FROM family_atlas", sys.conn)
        if not df.empty:
            c1, c2 = st.columns(2)
            c1.metric("家族年缴保费", f"¥{df['premium'].sum():,.2f}")
            c2.metric("覆盖成员数", f"{len(df['insured'].unique())} 人")
            st.bar_chart(df, x="insured", y="premium", color="category")

            st.subheader("🚨 智能风险预警")
            for m in df['insured'].unique():
                mdf = df[df['insured'] == m]
                if '重疾' not in mdf['category'].values and '重疾险' not in mdf['category'].values:
                    st.warning(f"成员【{m}】缺少重疾险配置，存在大病资金断裂风险。")
        else:
            st.info("暂无数据，请先在【拍照解析】中录入保单。")

    # ---------------- 模块3：车险理赔预判 ----------------
    with t3:
        st.subheader("车险事故智能向导")
        acc_text = st.text_area("请描述事故经过 (如：我变道时不小心刮蹭了直行的奥迪...)", height=150)
        if st.button("⚖️ 生成理赔建议", use_container_width=True):
            if acc_text and ds_key:
                with st.spinner("正在推演交通事故责任与赔付路径..."):
                    ans = sys.predict_auto_claim(acc_text)
                    st.markdown(f"> **AI 理赔意见：**\n{ans}")
            elif not ds_key:
                st.error("请配置 API Key")

    # ---------------- 模块4：条款智能问答 ----------------
    with t4:
        st.subheader("长文本条款 AI 律师")
        pdf_file = st.file_uploader("上传保险合同全文 (PDF格式)", type=['pdf'])
        question = st.text_input("向条款提问 (如：等待期是多久？既往症赔吗？)")

        if st.button("💡 查阅条款并解答", use_container_width=True):
            if pdf_file and question and ds_key:
                with st.spinner("正在提取 PDF 文本并思考..."):
                    # 使用 PyPDF2 提取文本
                    reader = PyPDF2.PdfReader(BytesIO(pdf_file.getvalue()))
                    full_text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])

                    if full_text:
                        ans = sys.qa_policy_terms(full_text, question)
                        st.success("解答完毕：")
                        st.write(ans)
                    else:
                        st.error("未能从 PDF 中提取出有效文本，请确认该 PDF 不是纯图片扫描件。")
            elif not ds_key:
                st.error("请配置 API Key")

    # ---------------- 模块5：系统设置 ----------------
    with t5:
        st.subheader("⚙️ 系统配置中心")
        # 输入框默认填入缓存中的 key
        new_key = st.text_input("DeepSeek API Key", type="password", value=st.session_state.api_key)

        if st.button("💾 保存配置", use_container_width=True):
            st.session_state.api_key = new_key
            st.success("配置已保存！各项 AI 功能已激活。")
            st.rerun()  # 刷新页面，让新的 Key 全局生效

        st.divider()
        st.caption("全功能版：包含图像增强、智能解析、家族看板、车险测算与条款问答。")


if __name__ == "__main__":
    main()