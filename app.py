import dashscope
import pandas as pd
import re
import time
import streamlit as st
import matplotlib.pyplot as plt
from rapidfuzz import fuzz, process
from dashscope import Generation
from collections import deque
import os
import matplotlib

try:
    # 尝试使用系统中可能有的中文字体
    system_fonts = matplotlib.font_manager.get_font_names()
    chinese_fonts = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi', 'FangSong', 'STXihei', 'STKaiti', 'STSong']

    available_font = None
    for font in chinese_fonts:
        if font in system_fonts:
            available_font = font
            break

    if available_font:
        plt.rcParams['font.sans-serif'] = [available_font]
        plt.rcParams['axes.unicode_minus'] = False
        print(f"使用中文字体: {available_font}")
    else:
        # 如果没有中文字体，尝试添加
        print("未找到系统中文字体，尝试其他方法...")
        # 使用默认字体，但可能显示为方块

except Exception as e:
    print(f"设置中文字体时出错: {e}")

# 设置页面配置
st.set_page_config(
    page_title="淘宝客服AI助手演示",
    page_icon="🤖",
    layout="wide"
)

# 初始化Session State
if 'history' not in st.session_state:
    st.session_state.history = deque(maxlen=3)  # 对话历史窗口
if 'all_conversations' not in st.session_state:
    st.session_state.all_conversations = []  # 完整对话记录
if 'knowledge_df' not in st.session_state:
    st.session_state.knowledge_df = None  # 统一知识库DataFrame
if 'rule_base' not in st.session_state:
    st.session_state.rule_base = None  # 规则库（仅用于意图识别）


def desensitize(text):
    """动态脱敏函数：部分遮蔽，保留信息可用性"""
    if not isinstance(text, str):
        return text

    # 1. 手机号 - 使用前后断言确保是独立的11位数字
    phone_pattern = r'(?<!\d)(1[3-9]\d{2})\d{4}(\d{3})(?!\d)'
    text = re.sub(phone_pattern, r'\1****\2', text)

    # 2. 身份证号 - 18位，使用前后断言
    id_card_pattern = r'(?<!\d)([1-9]\d{5})\d{8}([\dXx]{4})(?!\d)'
    text = re.sub(id_card_pattern, r'\1********\2', text)

    # 3. 订单号 - 可变长度，但至少7位，确保前后不是数字
    # 避免匹配到手机号或身份证号
    order_pattern = r'(?<!\d)(\d{3})\d+(\d{4})(?!\d)'
    text = re.sub(order_pattern, r'\1****\2', text)

    # 4. 邮政编码 - 6位数字，前后不是数字
    zip_code_pattern = r'(?<!\d)(\d{2})\d{2}(\d{2})(?!\d)'
    text = re.sub(zip_code_pattern, r'\1**\2', text)

    # 5. 邮箱 - 前后不是字母数字或@
    email_pattern = r'(?<![a-zA-Z0-9@])([a-zA-Z0-9_.+-]+)@([a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)(?![a-zA-Z0-9-.])'

    def email_replacer(match):
        username = match.group(1)
        domain = match.group(2)
        if len(username) > 2:
            return f'{username[:2]}***@{domain}'
        else:
            return f'{username}***@{domain}'

    text = re.sub(email_pattern, email_replacer, text)

    # 6. 地址 - 最后处理，使用更灵活的模式
    city_list = ['北京', '上海', '广州', '深圳', '杭州', '成都', '重庆', '武汉', '南京', '天津', '西安', '长沙', '沈阳',
                 '郑州', '济南', '青岛', '苏州', '无锡', '宁波', '东莞']
    city_str = '|'.join(city_list)

    # 改进的地址模式，匹配城市+详细地址直到遇到标点或结尾
    address_pattern = rf'(?P<city>{city_str})市?(?P<detail>[^，。！？；,\.!?;]*?(?:路|街|巷|号|弄|小区|幢|单元|室)[^，。！？；,\.!?;]*)'

    def address_replacer(match):
        city = match.group('city')
        return f'{city}[地址详情已遮蔽]'

    text = re.sub(address_pattern, address_replacer, text)

    return text


@st.cache_data
def load_knowledge_base(uploaded_file):
    """
    加载统一知识库Excel文件,知识库应包含`问题`、`问题类型`、`标准回答`三列
    """
    try:
        df = pd.read_excel(uploaded_file)
        required_columns = ['问题', '问题类型', '标准回答']
        for col in required_columns:
            if col not in df.columns:
                st.error(f"知识库文件必须包含'{col}'列")
                return None, None

        df = df.dropna(subset=['问题', '标准回答']).reset_index(drop=True)

        # 扩展后的规则库 - 意图路由器，引导系统去知识库中查找答案
        rule_base = {
            # 原有类别
            "发票咨询": {
                "patterns": ["发票", "开票", "专票", "普票", "税点", "开发票", "增值税", "抬头", "发票抬头"],
            },
            "物流查询": {
                "patterns": ["发货", "快递", "物流", "顺丰", "送达", "配送", "运输", "几天到", "发货时间", "快递单号",
                             "运费", "快递公司"],
            },
            "退货政策": {
                "patterns": ["退货", "退款", "退换货", "退货流程", "退货政策", "退货条件", "退货运费", "退货申请",
                             "退货怎么退"],
            },
            "售后政策": {
                "patterns": ["保修", "质保", "维修", "售后", "坏了", "保修期", "质保期", "维修服务", "售后支持",
                             "报修"],
            },

            # 新增类别
            "价格咨询": {
                "patterns": ["价格", "多少钱", "价", "优惠", "折扣", "便宜", "价位", "报价", "价格多少", "有优惠吗",
                             "价格优惠", "打折"],
            },
            "电机技术咨询": {
                "patterns": ["电机", "M0601", "M0602", "M1502", "M0603", "M0701", "M1505", "P1010",
                             "编码器", "减速器", "波特率", "CAN", "上位机", "电压", "扭矩", "转矩",
                             "电流", "转速", "PID", "位置环", "速度环", "电流环", "CANopen", "通信协议",
                             "例程", "代码", "固件", "驱动程序", "安装", "接线", "参数", "规格", "参数配置",
                             "电池", "电源", "电压范围", "供电", "功率", "力矩", "负载", "承重", "重量"],
            },
            "通用问答": {
                "patterns": ["你好", "您好", "hello", "hi", "早上好", "下午好", "晚上好", "在吗", "有人吗", "客服"],
            },
            "感谢与告别": {
                "patterns": ["谢谢", "感谢", "辛苦了", "再见", "拜拜", "下次见", "结束了", "好了", "没问题了"],
            }
        }

        return df, rule_base

    except Exception as e:
        st.error(f"知识库加载失败: {str(e)}")
        return None, None


def find_in_knowledge_base(user_query, knowledge_df):
    """
    系统核心查询函数 - 智能匹配版：平衡准确性和召回率
    """
    print(f"\n=== DEBUG find_in_knowledge_base 开始 ===")
    print(f"用户查询: {user_query}")
    
    if knowledge_df is None or knowledge_df.empty:
        print(f"DEBUG: 知识库为空")
        return None, None
    
    # ====== 第一步：强力拦截外观问题 ======
    # 只要包含这些关键词，就跳过知识库匹配
    appearance_keywords = [
        "颜色", "红色", "蓝色", "绿色", "黄色", "白色", "黑色", "灰色", 
        "外观", "样子", "外形", "形状", "长得", 
        "尺寸", "大小", "长", "宽", "高", 
        "材质", "材料", "塑料", "金属", 
        "重量", "重", "轻", "多重"
    ]
    
    # 检查是否包含外观关键词
    for keyword in appearance_keywords:
        if keyword in user_query:
            print(f"DEBUG: 发现外观关键词 '{keyword}'，跳过知识库匹配")
            return None, None
    
    # ====== 第二步：精确匹配 ======
    exact_match = knowledge_df[knowledge_df['问题'].str.strip().str.lower() == user_query.strip().lower()]
    if not exact_match.empty:
        answer = exact_match.iloc[0]['标准回答']
        print(f"DEBUG: 精确匹配成功，问题: {exact_match.iloc[0]['问题']}")
        return answer, exact_match.iloc[0].get('问题类型', '通用咨询')
    
    print(f"DEBUG: 精确匹配失败")
    
    # ====== 第三步：合并问题处理 ======
    # 检查是否是合并问题（包含"和"、"及"、"还有"等连接词）
    connectors = ["和", "及", "还有", "以及", "并且", "同时", "、"]
    has_connector = any(connector in user_query for connector in connectors)
    
    if has_connector:
        print(f"DEBUG: 检测到合并问题，尝试拆分处理")
        
        # 尝试根据连接词拆分问题
        found_answers = []
        
        # 检查各种连接词
        for connector in connectors:
            if connector in user_query:
                parts = [part.strip() for part in user_query.split(connector) if part.strip()]
                
                # 如果拆分成至少2部分，尝试分别匹配
                if len(parts) >= 2:
                    print(f"DEBUG: 按'{connector}'拆分为: {parts}")
                    
                    for part in parts:
                        # 为每个部分查找最佳匹配
                        part_matches = []
                        
                        # 1. 子串匹配
                        for idx, row in knowledge_df.iterrows():
                            question = row['问题'].strip().lower()
                            if part.lower() in question or question in part.lower():
                                part_matches.append((row['标准回答'], row.get('问题类型', '通用咨询'), 100))
                                break
                        
                        # 2. 模糊匹配
                        if not part_matches:
                            result = process.extractOne(
                                part,
                                knowledge_df['问题'].tolist(),
                                scorer=fuzz.token_set_ratio
                            )
                            
                            if result:
                                best_match, score, index = result
                                if score >= 50:  # 合并问题的部分匹配可以降低阈值
                                    matched_row = knowledge_df.iloc[index]
                                    part_matches.append((matched_row['标准回答'], matched_row.get('问题类型', '通用咨询'), score))
                        
                        if part_matches:
                            # 选择分数最高的
                            part_matches.sort(key=lambda x: x[2], reverse=True)
                            found_answers.append(part_matches[0][0])
                            print(f"DEBUG: 部分'{part}'匹配到答案")
        
        # 如果有找到多个答案，合并它们
        if len(found_answers) >= 2:
            print(f"DEBUG: 合并问题找到{len(found_answers)}个答案，进行合并")
            
            # 去重
            unique_answers = []
            for ans in found_answers:
                if ans not in unique_answers:
                    unique_answers.append(ans)
            
            if len(unique_answers) == 1:
                return unique_answers[0], "组合问题"
            else:
                # 组合多个答案
                combined_reply = "关于您的问题，分别回答如下：\n\n"
                for i, ans in enumerate(unique_answers, 1):
                    # 清理答案格式
                    clean_ans = ans.strip()
                    if not clean_ans.endswith(('。', '!', '?', '！', '？')):
                        clean_ans += '。'
                    combined_reply += f"{i}. {clean_ans}\n"
                
                return combined_reply, "组合问题"
        elif found_answers:
            # 只找到一个答案，直接返回
            return found_answers[0], "组合问题"
    
    # ====== 第四步：子串匹配（双向） ======
    # 只有当用户问题在知识库问题中是子串时才匹配，或者反过来
    for idx, row in knowledge_df.iterrows():
        question = row['问题'].strip().lower()
        user_q_lower = user_query.strip().lower()
        
        # 双向子串匹配
        if user_q_lower in question or question in user_q_lower:
            print(f"DEBUG: 子串匹配成功: {user_query} -> {question}")
            return row['标准回答'], row.get('问题类型', '通用咨询')
    
    print(f"DEBUG: 子串匹配失败")
    
    # ====== 第五步：智能模糊匹配（针对技术问题） ======
    # 检查是否是技术问题
    technical_keywords = ["电机", "M0601", "M0602", "M1502", "M0603", "M0701", "M1505", "P1010",
                         "编码器", "减速器", "波特率", "CAN", "上位机", "电压", "扭矩", "转矩",
                         "电流", "转速", "PID", "位置环", "速度环", "电流环", "CANopen", "通信协议",
                         "例程", "代码", "固件", "驱动程序", "安装", "接线", "参数", "规格"]
    
    is_technical_question = any(keyword in user_query for keyword in technical_keywords)
    
    if is_technical_question:
        print(f"DEBUG: 检测到技术问题，尝试模糊匹配")
        
        # 只对技术问题进行模糊匹配
        # rapidfuzz 返回三个值：(最佳匹配, 分数, 索引)
        result = process.extractOne(
            user_query,
            knowledge_df['问题'].tolist(),
            scorer=fuzz.token_set_ratio  # 使用token_set_ratio，对词序不敏感
        )
        
        if result:
            best_match, score, index = result
            print(f"DEBUG: 模糊匹配结果: {best_match}")
            print(f"DEBUG: 匹配分数: {score}")
            print(f"DEBUG: 匹配索引: {index}")
            
            # 对于技术问题，降低阈值到50
            if score >= 50:  # 降低阈值到50，提高召回率
                matched_row = knowledge_df.iloc[index]  # 直接使用索引获取行
                
                # 验证匹配的相关性
                # 检查匹配到的问题是否也是技术问题
                matched_is_technical = any(keyword in best_match for keyword in technical_keywords)
                
                if matched_is_technical:
                    print(f"DEBUG: 模糊匹配成功，返回知识库答案")
                    return matched_row['标准回答'], matched_row.get('问题类型', '通用咨询')
                else:
                    print(f"DEBUG: 匹配到非技术问题，拒绝返回")
            else:
                print(f"DEBUG: 模糊匹配分数不足 {score} < 50")
        else:
            print(f"DEBUG: 模糊匹配未找到结果")
    
    # 没有找到匹配
    print(f"DEBUG: 所有匹配方法都失败")
    return None, None

def rule_engine(user_query, knowledge_df):
    """
    识别意图,并尝试从对应类型的知识库中获取答案
    """
    start_time = time.time()
    print(f"\n=== DEBUG rule_engine 开始 ===")
    print(f"用户查询: {user_query}")
    
    user_query_lower = user_query.lower()
    rule_base = st.session_state.rule_base
    
    # ==== 新增：特殊处理外观属性问题 ====
    # 定义外观属性关键词（更全面）
    appearance_keywords = [
        # 颜色相关
        "颜色", "红色", "蓝色", "绿色", "黄色", "白色", "黑色", "灰色", "银色", "金色",
        "什么颜色", "颜色是", "啥颜色", "颜色的", "色",
        # 外观相关
        "外观", "样子", "外形", "形状", "长得", "长什么样", "好看", "漂亮", "颜值",
        "外观设计", "外观是", "外观怎么样",
        # 尺寸相关
        "尺寸", "大小", "长", "宽", "高", "厚度", "直径", "体积", "尺寸多大",
        "多长", "多宽", "多高", "多大尺寸", "大小是",
        # 材质相关
        "材质", "材料", "塑料", "金属", "铝合金", "不锈钢", "铁", "钢",
        "什么材质", "什么材料", "用的什么",
        # 重量相关
        "重量", "重", "轻", "多重", "几公斤", "多少克", "重量多少"
    ]
    
    # 检查是否包含外观属性关键词
    has_appearance_keyword = False
    matched_keywords = []
    for keyword in appearance_keywords:
        if keyword in user_query:
            has_appearance_keyword = True
            matched_keywords.append(keyword)
    
    print(f"是否包含外观关键词: {has_appearance_keyword}")
    if has_appearance_keyword:
        print(f"匹配到的外观关键词: {matched_keywords}")
    
    # 关键修改：只要包含外观关键词，就强制使用AI处理
    if has_appearance_keyword:
        # 但需要排除技术上下文（比如"红色指示灯"）
        technical_contexts = ["指示灯", "LED", "灯", "报警", "故障", "状态", "显示", "信号", 
                             "电压", "电流", "转速", "扭矩", "编码器", "减速器", "通信"]
        has_technical_context = any(context in user_query for context in technical_contexts)
        
        print(f"是否包含技术上下文: {has_technical_context}")
        
        # 如果没有技术上下文，直接强制使用AI
        if not has_technical_context:
            end_time = time.time()
            print(f"DEBUG: 外观问题，强制使用AI处理")
            return {
                "source": "规则引擎",
                "intent": "外观属性咨询",
                "reply": None,  # 返回None，让AI处理
                "latency": end_time - start_time,
                "score": 0,
                "status": "failed"  # 标记为失败，让后续流程处理
            }
    
    # ==== 原有意图识别逻辑 ====
    detected_intent = None
    for intent, config in rule_base.items():
        if any(word in user_query_lower for word in config["patterns"]):
            detected_intent = intent
            print(f"规则引擎识别到意图: {intent}")
            break

    # 特殊处理：通用问答和感谢告别
    if detected_intent == "通用问答":
        end_time = time.time()
        print(f"DEBUG: 通用问答，使用预设回复")
        return {
            "source": "系统预设",
            "intent": "通用问答",
            "reply": "您好！我是本末科技的智能客服，很高兴为您服务。有什么可以帮助您的吗？",
            "latency": end_time - start_time,
            "score": 100,
            "status": "success"
        }
    elif detected_intent == "感谢与告别":
        end_time = time.time()
        print(f"DEBUG: 感谢告别，使用预设回复")
        return {
            "source": "系统预设",
            "intent": "感谢与告别",
            "reply": "不客气，这是我应该做的！如有其他问题随时联系我，祝您生活愉快！",
            "latency": end_time - start_time,
            "score": 100,
            "status": "success"
        }

    # 无论是否识别出具体意图，都先在知识库中全局查找
    print(f"调用 find_in_knowledge_base...")
    reply, detected_type = find_in_knowledge_base(user_query, knowledge_df)

    end_time = time.time()

    if reply:
        # 成功从知识库中找到答案
        # 使用检测到的问题类型作为意图，如果未指定则使用规则引擎检测的意图
        intent_used = detected_type if detected_type else (detected_intent if detected_intent else "知识库匹配")
        print(f"DEBUG: 知识库匹配成功，返回答案")
        print(f"匹配到的问题类型: {detected_type}")
        print(f"使用的意图: {intent_used}")
        return {
            "source": f"知识库 ({intent_used})",
            "intent": intent_used,
            "reply": reply,
            "latency": end_time - start_time,
            "score": 100,
            "status": "success"
        }
    else:
        # 知识库中未找到答案
        print(f"DEBUG: 知识库未找到答案")
        return {
            "source": "规则引擎",
            "intent": detected_intent if detected_intent else "未识别",
            "reply": None,
            "latency": end_time - start_time,
            "score": 0,
            "status": "failed"
        }

def ai_enhancement_with_knowledge(user_query, history_window, knowledge_df):
    """
    增强版AI生成回复：结合知识库中的相关信息，生成简洁回答
    """
    start_time = time.time()
    
    # 1. 检查是否是外观属性问题
    is_appearance_question = False
    appearance_keywords = [
        "颜色", "红色", "蓝色", "绿色", "黄色", "白色", "黑色", "外观", "样子", 
        "外形", "形状", "长得", "尺寸", "大小", "长", "宽", "高", "材质", "材料",
        "重量", "多重", "重"
    ]
    
    if any(keyword in user_query for keyword in appearance_keywords):
        is_appearance_question = True
    
    # 2. 从知识库中检索相关上下文
    relevant_knowledge = ""
    if knowledge_df is not None and not knowledge_df.empty:
        # 尝试查找最相关的问题
        best_answer, _ = find_in_knowledge_base(user_query, knowledge_df)
        if best_answer:
            relevant_knowledge = f"知识库标准答案：{best_answer}\n\n"
    
    # 3. 构建Prompt - 特别要求简洁回答
    history_text = "\n".join([f"用户：{q}\n客服:{a}" for q, a in history_window])
    
    # 根据问题类型调整Prompt
    technical_keywords = ["电机", "M0601", "M0602", "M1502", "编码器", "减速器", "CAN", 
                         "上位机", "电压", "代码", "例程", "通信", "波特率"]
    
    is_technical = any(keyword in user_query for keyword in technical_keywords)
    
    if is_technical and relevant_knowledge:
        # 技术问题且有知识库答案时，生成简洁回答
        full_prompt = f"""你是一个专业的机器人产品淘宝客服AI助手。

**重要指令**：
1. 下面提供了知识库中的标准答案
2. 如果知道确切答案，请准确、简洁地回答
3. 如果不知道确切答案，请说"抱歉，我暂时无法回答这个问题，建议您联系客服或查看产品说明书"
4. 绝对不要编造参数、规格、公司地址等具体信息，尤其是知识库没有提到的信息。不要猜测
5. 如果用户问的是技术参数，直接回答参数

**知识库标准答案**：
{relevant_knowledge}

**当前用户问题**：
{user_query}

请生成简洁、专业的客服回复（最好在50字以内）："""
    elif is_appearance_question:
        # 外观问题
        full_prompt = f"""你是一个专业的机器人产品淘宝客服AI助手。

用户问了一个关于产品外观/颜色/尺寸的问题，但知识库中没有相关信息。

**当前用户问题**：
{user_query}

请根据常识生成简短回复（30字以内），如果不知道确切信息，可以说明情况并提供帮助方式。"""
    else:
        # 其他问题
        full_prompt = f"""你是一个专业的机器人产品淘宝客服AI助手。

**重要指令**：
1. 请优先参考下面的知识库信息
2. 如果知识库信息能回答用户问题，请基于知识库信息生成简洁回复
3. 如果知识库信息不完整，可以补充你的专业知识。
4. 但是涉及到你不确定且知识库完全没出现的内容时，请说抱歉我不知道，建议您联系客服或查看公司官网或产品说明书。
5. 注意不要泄露任何隐私信息
6. 保持回答简洁明了

**知识库参考信息**：
{relevant_knowledge if relevant_knowledge else "（暂无相关参考信息）"}

**对话历史(最近3轮)**：
{history_text if history_text else "（暂无历史对话）"}

**当前用户问题**：
{user_query}

请生成简洁、友好的客服回复："""

    try:
        # 获取API密钥
        api_key = st.session_state.get('api_key', '')
        if not api_key:
            api_key = os.getenv('DASHSCOPE_API_KEY', '')
            if not api_key:
                return {
                    "source": "AI模型",
                    "intent": "未识别",
                    "reply": "⚠️ 未配置API密钥，请在侧边栏设置",
                    "latency": time.time() - start_time,
                    "status": "failed"
                }
        
        response = Generation.call(
            model="qwen-plus",
            prompt=full_prompt,
            temperature=0.3,
            api_key=api_key
        )
        
        end_time = time.time()
        
        if response.status_code == 200:
            reply = response.output.text
            reply = desensitize(reply)
                        
            return {
                "source": "AI模型" + ("（外观咨询）" if is_appearance_question else "（增强版）"),
                "intent": "外观属性咨询" if is_appearance_question else "未识别",
                "reply": reply,
                "latency": end_time - start_time,
                "status": "success"
            }
        else:
            return {
                "source": "AI模型",
                "intent": "未识别",
                "reply": f"请求失败，请稍后再试 (错误码: {response.status_code})",
                "latency": end_time - start_time,
                "status": "failed"
            }
    except Exception as e:
        end_time = time.time()
        return {
            "source": "AI模型",
            "intent": "未识别",
            "reply": f"API调用异常: {str(e)[:50]}...",
            "latency": end_time - start_time,
            "status": "failed"
        }

def process_query(user_query):
    """
    知识库优先,匹配失败时调用增强版AI模型（带知识库上下文）
    """
    print(f"\n=== DEBUG process_query 开始 ===")
    print(f"用户查询: {user_query}")
    
    knowledge_df = st.session_state.knowledge_df
    
    # 直接使用规则引擎
    rule_result = rule_engine(user_query, knowledge_df)
    
    print(f"DEBUG: rule_engine 返回状态: {rule_result['status']}")
    print(f"DEBUG: rule_engine 返回source: {rule_result['source']}")
    
    if rule_result["status"] == "success":
        print(f"DEBUG: 使用知识库/预设回复")
        # 记录到对话历史
        st.session_state.history.appendleft((user_query, rule_result["reply"]))
        st.session_state.all_conversations.append({
            "query": user_query,
            "reply": rule_result["reply"],
            "source": rule_result["source"],
            "time": time.strftime("%H:%M:%S"),
            "latency": rule_result["latency"]
        })
        return rule_result
    else:
        print(f"DEBUG: 调用AI增强版")
        # 知识库无法回答，调用增强版AI
        ai_result = ai_enhancement_with_knowledge(
            user_query, 
            st.session_state.history,
            knowledge_df
        )
        
        # 记录到对话历史
        st.session_state.history.appendleft((user_query, ai_result["reply"]))
        st.session_state.all_conversations.append({
            "query": user_query,
            "reply": ai_result["reply"],
            "source": ai_result["source"],
            "time": time.strftime("%H:%M:%S"),
            "latency": ai_result["latency"]
        })
        return ai_result


def generate_statistics_chart():
    """生成简单的统计图表"""
    if len(st.session_state.all_conversations) == 0:
        return None

    df = pd.DataFrame(st.session_state.all_conversations)

    # 创建图表
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 触发来源分布
    source_counts = df['source'].value_counts()
    axes[0].pie(source_counts.values, labels=source_counts.index, autopct='%1.1f%%', startangle=90)
    axes[0].set_title('触发来源分布')

    # 响应时间趋势
    if len(df) > 1:
        df['index'] = range(len(df))
        axes[1].plot(df['index'], df['latency'], marker='o')
        axes[1].set_xlabel('对话序号')
        axes[1].set_ylabel('响应时间(秒)')
        axes[1].set_title('响应时间趋势')
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# Streamlit界面
def main():
    st.title("🤖 机器人客服AI助手演示系统")
    st.markdown("---")

    # 侧边栏 - 配置区域
    with st.sidebar:
        st.header("⚙️ 系统配置")

        # API密钥设置
        with st.expander("API配置"):
            # 获取当前session_state中的API密钥，如果不存在则显示空字符串
            current_api_key = st.session_state.get('api_key', '')
            api_key = st.text_input("通义千问API密钥",
                                    type="password",
                                    value=current_api_key,
                                    help="输入你的阿里云DashScope API密钥")

            # 当用户输入API密钥后，保存到session_state
            if api_key and api_key != current_api_key:
                st.session_state['api_key'] = api_key
                st.success("API密钥已更新!")

            # 添加一个测试连接按钮
            if st.button("测试API连接"):
                if st.session_state.get('api_key'):
                    dashscope.api_key = st.session_state['api_key']
                    try:
                        # 简单测试调用
                        test_response = Generation.call(
                            model="qwen-plus",
                            prompt="你好",
                            temperature=0.1
                        )
                        if test_response.status_code == 200:
                            st.success("API连接成功!")
                        else:
                            st.error(f"API连接失败: {test_response.message}")
                    except Exception as e:
                        st.error(f"连接异常: {str(e)}")
                else:
                    st.warning("请先输入API密钥")

        # 数据上传
        st.subheader("📊 数据上传")
        uploaded_file = st.file_uploader("上传知识库Excel文件", type=['xlsx'],
                                         help="请确保文件包含'问题'、'问题类型'、'标准回答'三列")

        if uploaded_file is not None:
            if st.button("加载知识库"):
                with st.spinner("正在加载知识库..."):
                    # 调用更新后的加载函数，现在返回两个值
                    df, rule_base = load_knowledge_base(uploaded_file)
                    if df is not None:
                        # 更新Session State变量名
                        st.session_state.knowledge_df = df
                        st.session_state.rule_base = rule_base
                        st.success(f"✅ 成功加载 {len(df)} 条知识记录")

                        # 显示问题类型分布，体现新架构优势
                        if '问题类型' in df.columns:
                            type_counts = df['问题类型'].value_counts()
                            type_info = ", ".join([f"{k}({v}条)" for k, v in type_counts.items()])
                            st.info(f"**问题类型分布:** {type_info}")

                            # 显示规则库覆盖情况
                            rule_categories = list(rule_base.keys())
                            st.info(f"**规则库覆盖:** {len(rule_categories)}个意图类别")

        # 系统状态 - 更新变量名
        st.subheader("📈 系统状态")
        st.metric("对话总数", len(st.session_state.all_conversations))
        st.metric("历史窗口大小", len(st.session_state.history))
        if st.session_state.knowledge_df is not None:
            st.metric("知识库条目", len(st.session_state.knowledge_df))
        if st.session_state.rule_base is not None:
            st.metric("规则库类别", len(st.session_state.rule_base))

        # 清空对话按钮
        if st.button("清空对话历史"):
            st.session_state.history.clear()
            st.session_state.all_conversations.clear()
            st.success("对话历史已清空")

    # 主界面 - 两列布局
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("💬 客服工作台")

        # 初始化 session_state
        if 'user_query' not in st.session_state:
            st.session_state.user_query = ""
        if 'query_submitted' not in st.session_state:
            st.session_state.query_submitted = False

        # 示例问题列表 - 按类别分组
        examples_by_category = {
            "电机技术咨询": [
                "M0601C电机带减速器吗?",
                "M0603C电机支持CAN通信吗?",
                "电机可以用24V电压吗?",
                "有代码例程和上位机吗?"
            ],
            "物流查询": [
                "什么时候发货？",
                "快递几天能到?",
                "发什么快递？",
                "运费怎么算？"
            ],
            "发票咨询": [
                "可以开发票吗？",
                "可以开专票吗？",
                "发票怎么开？",
                "发票开错了可以重开吗？"
            ],
            "价格与售后": [
                "产品有优惠吗？能便宜点吗?",
                "怎么申请退货？",
                "保修期多久?",
                "运费可以便宜吗？"
            ]
        }

        # 显示示例问题
        st.markdown("**快速提问（点击直接使用）**")

        # 创建一个容器来显示示例按钮
        example_container = st.container()

        # 使用tab显示不同类别
        tabs = example_container.tabs(list(examples_by_category.keys()))

        for tab_idx, (category, examples) in enumerate(examples_by_category.items()):
            with tabs[tab_idx]:
                cols = st.columns(2)
                for idx, example in enumerate(examples):
                    col_idx = idx % 2
                    with cols[col_idx]:
                        # 定义按钮点击回调函数
                        def set_example_query(example=example):
                            st.session_state.user_query = example
                            st.session_state.query_submitted = True

                        btn_text = f"📌 {example[:20]}..." if len(example) > 20 else f"📌 {example}"
                        st.button(
                            btn_text,
                            key=f"ex_btn_{category}_{idx}",
                            on_click=set_example_query,
                            use_container_width=True
                        )

        st.markdown("---")

        # 使用 form 来管理输入和提交
        with st.form(key="query_form", clear_on_submit=False):
            # 显示当前已选中的问题
            current_query = st.session_state.user_query
            query_display = st.text_input(
                "已选问题:",
                value=current_query,
                disabled=True,
                key="query_display"
            )

            # 允许用户编辑
            user_query = st.text_area(
                "编辑或输入新问题：",
                value=current_query,
                placeholder="例如:M0601C电机带减速器吗?编码器是绝对式的吗?",
                height=100,
                key="user_input"
            )

            # 提交按钮
            submit_col1, submit_col2 = st.columns([2, 1])
            with submit_col1:
                submitted = st.form_submit_button("🚀 获取AI回复", type="primary", use_container_width=True)
            with submit_col2:
                clear_clicked = st.form_submit_button("🗑️ 清空", type="secondary", use_container_width=True)

            # 当清空按钮被点击时
            if clear_clicked:
                st.session_state.user_query = ""
                # 这里不需要 rerun，因为清空后，表单重新渲染时会使用空值

            # 当表单提交时
            if submitted and user_query:
                # 更新 session_state
                st.session_state.user_query = user_query
                st.session_state.query_submitted = True

        # 当 query_submitted 为 True 时，处理查询
        if st.session_state.query_submitted and st.session_state.user_query:
            # 重置提交状态
            st.session_state.query_submitted = False

            if st.session_state.knowledge_df is None:
                st.warning("⚠️ 请先上传知识库数据")
            else:
                with st.spinner("正在生成回复..."):
                    result = process_query(st.session_state.user_query)

                    # 显示结果
                    st.markdown("---")
                    
                    # ============ 错误处理部分 ============
                    if result["status"] == "failed":
                        st.error(f"⚠️ 系统处理遇到问题: {result['reply']}")
                        
                        # 提供备选方案
                        st.markdown("### 🔍 建议尝试以下方法:")
                        st.markdown("1. 将复杂问题拆分为多个简单问题询问")
                        st.markdown("2. 检查API密钥是否正确配置")
                        st.markdown("3. 稍后重试或联系技术支持")
                        
                        # 如果知识库有相关内容，尝试提供一些可能的答案
                        if st.session_state.knowledge_df is not None:
                            # 尝试从知识库中找到部分相关答案
                            query_lower = st.session_state.user_query.lower()
                            related_questions = []
                            
                            # 检查常见关键词
                            keywords = ["代码", "例程", "上位机", "电机", "控制", "软件"]
                            for keyword in keywords:
                                if keyword in query_lower:
                                    matches = st.session_state.knowledge_df[
                                        st.session_state.knowledge_df['问题'].str.contains(keyword, case=False, na=False)
                                    ]
                                    if not matches.empty:
                                        for _, row in matches.head(2).iterrows():
                                            related_questions.append({
                                                "问题": row['问题'],
                                                "答案": row['标准回答']
                                            })
                            
                            if related_questions:
                                st.markdown("### 📚 知识库相关问答:")
                                for i, item in enumerate(related_questions[:3], 1):
                                    with st.expander(f"相关问答 {i}: {item['问题'][:30]}..."):
                                        st.markdown(f"**问题:** {item['问题']}")
                                        st.markdown(f"**答案:** {item['答案']}")
                        
                        # 结束当前处理
                        st.stop()

                    # ============ 正常结果显示 ============
                    with st.container():
                        st.markdown("### 🤖 AI回复建议")

                        # 显示来源标签
                        source_text = result["source"]
                        if "知识库" in source_text:
                            source_color = "#4CAF50"  # 绿色
                            icon = "📚"
                        elif "AI模型" in source_text:
                            source_color = "#2196F3"  # 蓝色
                            icon = "🤖"
                        elif "系统预设" in source_text:
                            source_color = "#9C27B0"  # 紫色
                            icon = "⚙️"
                        else:
                            source_color = "#FF9800"  # 橙色
                            icon = "🔧"

                        # 显示意图标签
                        intent_text = result.get("intent", "未识别")
                        intent_colors = {
                            "发票咨询": "#FF5722",
                            "物流查询": "#3F51B5",
                            "退货政策": "#E91E63",
                            "售后政策": "#009688",
                            "价格咨询": "#FF9800",
                            "电机技术咨询": "#795548",
                            "通用问答": "#9C27B0",
                            "感谢与告别": "#607D8B",
                            "未识别": "#9E9E9E"
                        }
                        intent_color = intent_colors.get(intent_text, "#9E9E9E")

                        col_source, col_intent, col_time = st.columns([2, 2, 1])
                        with col_source:
                            st.markdown(f"""
                            <div style="background-color:{source_color}; color:white; padding:5px 10px; 
                                        border-radius:5px; display:inline-block; margin-bottom:10px;">
                                {icon} {source_text}
                            </div>
                            """, unsafe_allow_html=True)
                        with col_intent:
                            st.markdown(f"""
                            <div style="background-color:{intent_color}; color:white; padding:5px 10px; 
                                        border-radius:5px; display:inline-block; margin-bottom:10px;">
                                🏷️ {intent_text}
                            </div>
                            """, unsafe_allow_html=True)
                        with col_time:
                            st.markdown(f"""
                            <div style="background-color:#616161; color:white; padding:5px 10px; 
                                        border-radius:5px; display:inline-block; margin-bottom:10px;">
                                ⏱️ {result["latency"]:.2f}秒
                            </div>
                            """, unsafe_allow_html=True)

                        # 显示回复内容
                        st.markdown(f"""
                        <div style="background-color:#f5f5f5; padding:15px; border-radius:5px; 
                                    border-left:4px solid {source_color}; margin:10px 0;">
                            {result["reply"]}
                        </div>
                        """, unsafe_allow_html=True)

                        # 一键复制按钮
                        st.code(result["reply"], language=None)

                        # 提示信息
                        if "知识库" in source_text:
                            st.caption("✅ 此回复来自知识库标准答案，准确可靠")
                        elif "AI模型" in source_text:
                            st.caption("🤖 此回复由AI生成，请仔细核对")
                        elif "系统预设" in source_text:
                            st.caption("⚙️ 此回复来自系统预设模板")
                        
                        # 添加用户反馈功能
                        st.markdown("---")
                        st.subheader("💬 反馈这个回答")
                        
                        col_fb1, col_fb2, col_fb3 = st.columns(3)
                        with col_fb1:
                            if st.button("👍 回答准确", use_container_width=True):
                                st.success("感谢您的反馈！")
                        with col_fb2:
                            if st.button("👎 回答不准确", use_container_width=True):
                                st.error("抱歉回答有误，我们会改进！")
                        with col_fb3:
                            if st.button("🤔 不确定", use_container_width=True):
                                st.info("感谢反馈，我们会检查这个问题。")

        # 对话历史
        st.markdown("---")
        st.subheader("📜 对话历史")

        if len(st.session_state.all_conversations) > 0:
            for i, conv in enumerate(st.session_state.all_conversations[-5:]):
                with st.expander(f"{conv['time']} - {conv['query'][:30]}..."):
                    col_a, col_b = st.columns([3, 1])
                    with col_a:
                        st.markdown(f"**用户问题:** {conv['query']}")
                        st.markdown(f"**客服回复:** {conv['reply']}")
                    with col_b:
                        source_text = conv['source']
                        if "知识库" in source_text:
                            source_badge = "🟢 知识库"
                        elif "AI模型" in source_text:
                            source_badge = "🔵 AI生成"
                        elif "系统预设" in source_text:
                            source_badge = "🟣 系统预设"
                        else:
                            source_badge = f"🟠 {source_text}"
                        st.caption(f"来源: {source_badge}")
                        st.caption(f"耗时: {conv['latency']:.2f}秒")
                        
                        # 添加删除按钮
                        if st.button(f"🗑️ 删除", key=f"delete_{i}"):
                            # 从对话历史中删除
                            del st.session_state.all_conversations[i]
                            st.rerun()
        else:
            st.info("暂无对话历史，请先提问")

    with col2:
        st.subheader("📊 系统信息")

        # 知识库状态
        if st.session_state.knowledge_df is not None:
            df = st.session_state.knowledge_df
            st.success(f"✅ 知识库已加载")
            st.metric("知识条目", len(df))

            # 显示知识库统计信息
            with st.expander("📋 知识库详情"):
                # 问题类型分布
                if '问题类型' in df.columns:
                    st.write("**问题类型分布:**")
                    type_counts = df['问题类型'].value_counts()
                    for type_name, count in type_counts.items():
                        # 创建水平条形图效果
                        percent = count / len(df) * 100
                        st.progress(percent / 100, text=f"{type_name}: {count}条 ({percent:.1f}%)")
                else:
                    st.write("知识库未标注问题类型")

                # 示例问题展示
                st.write("**示例问题:**")
                # 优先展示不同类型的问题
                sample_size = min(5, len(df))
                if '问题类型' in df.columns and df['问题类型'].nunique() > 1:
                    # 尝试从不同类别取样
                    samples = []
                    for type_name in df['问题类型'].unique():
                        type_samples = df[df['问题类型'] == type_name].head(1)
                        samples.extend(type_samples['问题'].tolist())
                    samples = samples[:sample_size]
                else:
                    samples = df['问题'].sample(sample_size).tolist()

                for q in samples:
                    st.caption(f"• {q[:25]}..." if len(q) > 25 else f"• {q}")

                # 显示规则库信息
                if st.session_state.rule_base is not None:
                    st.write("**规则库覆盖类别:**")
                    rule_categories = list(st.session_state.rule_base.keys())
                    for category in rule_categories:
                        pattern_count = len(st.session_state.rule_base[category]["patterns"])
                        st.caption(f"• {category} ({pattern_count}个关键词)")
                        
                # 添加知识库导出功能
                st.markdown("---")
                if st.button("📥 导出知识库统计", use_container_width=True):
                    # 创建统计DataFrame
                    stats_df = pd.DataFrame({
                        '指标': ['总问题数', '问题类型数', '平均回答长度'],
                        '数值': [
                            len(df),
                            df['问题类型'].nunique() if '问题类型' in df.columns else 0,
                            df['标准回答'].str.len().mean()
                        ]
                    })
                    st.dataframe(stats_df, use_container_width=True)
        else:
            st.warning("📁 等待加载知识库")
            st.info("请上传包含三列(问题、标准回答、问题类型)的Excel文件")

        # 脱敏演示
        st.markdown("---")
        st.subheader("🔒 脱敏演示")

        # 使用session_state存储测试文本
        if 'test_text' not in st.session_state:
            st.session_state.test_text = "我的手机是15766265746, 地址是杭州市西湖区文三路"

        test_text = st.text_area(
            "输入测试文本:",
            value=st.session_state.test_text,
            height=100,
            key="test_input"
        )

        # 添加一个按钮来触发脱敏
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("运行脱敏测试", type="primary"):
                st.session_state.test_text = test_text

                # 直接测试脱敏函数
                st.markdown("### 测试结果")

                # 1. 先显示原始文本
                st.markdown("**原始文本:**")
                st.code(test_text)

                # 2. 调用脱敏函数
                result = desensitize(test_text)

                # 3. 显示脱敏结果
                st.markdown("**脱敏结果:**")
                st.code(result)

                # 4. 显示对比
                if result != test_text:
                    st.success("✅ 脱敏成功!")
                else:
                    st.error("❌ 脱敏失败! 文本没有变化。")

        # 统计图表
        st.markdown("---")
        st.subheader("📈 性能统计")

        if len(st.session_state.all_conversations) > 0:
            fig = generate_statistics_chart()
            if fig:
                st.pyplot(fig)

            # 简单统计
            df_stats = pd.DataFrame(st.session_state.all_conversations)
            if not df_stats.empty:
                avg_latency = df_stats['latency'].mean()

                # 新的统计逻辑：知识库命中 vs AI生成 vs 系统预设
                kb_count = len(df_stats[df_stats['source'].str.contains('知识库')])
                ai_count = len(df_stats[df_stats['source'].str.contains('AI模型')])
                sys_count = len(df_stats[df_stats['source'].str.contains('系统预设')])
                other_count = len(df_stats) - kb_count - ai_count - sys_count

                st.metric("平均响应时间", f"{avg_latency:.2f}秒")

                # 显示回答来源分布
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("知识库命中", kb_count)
                with col2:
                    st.metric("AI生成", ai_count)
                with col3:
                    st.metric("系统预设", sys_count)

                # 计算命中率
                if len(df_stats) > 0:
                    hit_rate = (kb_count + sys_count) / len(df_stats) * 100
                    st.progress(hit_rate / 100, text=f"知识库+预设命中率: {hit_rate:.1f}%")
                    
                # 添加性能建议
                with st.expander("📊 性能分析建议"):
                    if avg_latency > 2.0:
                        st.warning("⚠️ 平均响应时间较长，建议:")
                        st.markdown("""
                        1. 检查API网络连接
                        2. 考虑使用本地缓存
                        3. 优化知识库匹配算法
                        """)
                    else:
                        st.success("✅ 响应时间正常")
                        
                    if hit_rate < 50:
                        st.warning(f"⚠️ 知识库命中率较低 ({hit_rate:.1f}%)，建议:")
                        st.markdown("""
                        1. 扩充知识库内容
                        2. 优化关键词匹配规则
                        3. 添加更多示例问题
                        """)
                    else:
                        st.success(f"✅ 知识库命中率良好 ({hit_rate:.1f}%)")

    # 页脚
    st.markdown("---")
    st.caption("💡 提示:这是一个演示系统,回复仅供参考。技术参数类问题优先从知识库匹配,其他问题由AI生成。")
    st.caption("⚠️ 实际使用时请确保数据安全并遵守平台规则。")


if __name__ == "__main__":
    main()


