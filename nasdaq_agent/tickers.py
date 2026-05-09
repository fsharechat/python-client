# 纳斯达克100成分股列表（2025年）
NASDAQ100_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "TSLA", "AVGO", "COST",
    "NFLX", "AMD", "ASML", "LIN", "ADBE", "QCOM", "INTU", "AMAT", "BKNG", "ISRG",
    "VRTX", "TXN", "ADP", "PANW", "MU", "GILD", "LRCX", "ADI", "REGN", "KLAC",
    "SNPS", "CDNS", "MDLZ", "SBUX", "INTC", "CTAS", "ORLY", "NXPI", "MRVL", "CEG",
    "PYPL", "CRWD", "WDAY", "CSX", "ABNB", "PAYX", "DXCM", "IDXX", "ODFL", "ROST",
    "FAST", "VRSK", "PCAR", "AEP", "CHTR", "FANG", "MNST", "BIIB", "ILMN", "TEAM",
    "MELI", "PDD", "GEHC", "DDOG", "ARM", "ZS", "ROP", "KDP", "CPRT", "DLTR",
    "CSGP", "EXC", "XEL", "BKR", "ANSS", "CTSH", "MCHP", "EBAY", "EA", "ON",
    "MRNA", "WBD", "ENPH", "ALGN", "KHC", "TTWO", "FTNT", "SIRI", "GFS", "APP",
    "PLTR", "AXON", "SMCI", "DASH", "COIN", "RBLX", "TTD", "ZM", "OKTA", "MTCH",
]

# ticker → 板块中文名
NASDAQ100_SECTOR_MAP: dict[str, str] = {
    # 半导体
    "NVDA": "半导体", "AMD": "半导体", "AVGO": "半导体", "QCOM": "半导体",
    "INTC": "半导体", "TXN": "半导体", "AMAT": "半导体", "LRCX": "半导体",
    "KLAC": "半导体", "NXPI": "半导体", "MRVL": "半导体", "MCHP": "半导体",
    "ADI": "半导体", "ON": "半导体", "MU": "半导体", "ASML": "半导体",
    "ARM": "半导体", "GFS": "半导体", "SNPS": "半导体", "CDNS": "半导体",
    # 大型科技
    "AAPL": "大型科技", "MSFT": "大型科技", "GOOGL": "大型科技",
    "GOOG": "大型科技", "META": "大型科技", "AMZN": "大型科技",
    # 软件/SaaS
    "ADBE": "软件/SaaS", "INTU": "软件/SaaS", "PANW": "软件/SaaS",
    "CRWD": "软件/SaaS", "WDAY": "软件/SaaS", "DDOG": "软件/SaaS",
    "ZS": "软件/SaaS", "FTNT": "软件/SaaS", "OKTA": "软件/SaaS",
    "TEAM": "软件/SaaS", "CTSH": "软件/SaaS", "ROP": "软件/SaaS",
    "ANSS": "软件/SaaS", "CSGP": "软件/SaaS",
    # 互联网/电商
    "BKNG": "互联网/电商", "ABNB": "互联网/电商", "EBAY": "互联网/电商",
    "MELI": "互联网/电商", "PDD": "互联网/电商", "DASH": "互联网/电商",
    "RBLX": "互联网/电商", "ZM": "互联网/电商", "MTCH": "互联网/电商",
    "TTD": "互联网/电商", "APP": "互联网/电商",
    # 医疗健康
    "ISRG": "医疗健康", "VRTX": "医疗健康", "GILD": "医疗健康",
    "REGN": "医疗健康", "BIIB": "医疗健康", "IDXX": "医疗健康",
    "DXCM": "医疗健康", "ILMN": "医疗健康", "MRNA": "医疗健康",
    "ALGN": "医疗健康", "GEHC": "医疗健康",
    # 消费/零售
    "COST": "消费/零售", "SBUX": "消费/零售", "ORLY": "消费/零售",
    "DLTR": "消费/零售", "ROST": "消费/零售", "MDLZ": "消费/零售",
    "MNST": "消费/零售", "KDP": "消费/零售", "KHC": "消费/零售",
    # 媒体/娱乐
    "NFLX": "媒体/娱乐", "CHTR": "媒体/娱乐", "WBD": "媒体/娱乐",
    "SIRI": "媒体/娱乐", "EA": "媒体/娱乐", "TTWO": "媒体/娱乐",
    # 工业/物流
    "ADP": "工业/物流", "PAYX": "工业/物流", "PCAR": "工业/物流",
    "ODFL": "工业/物流", "CSX": "工业/物流", "FAST": "工业/物流",
    "VRSK": "工业/物流", "CTAS": "工业/物流", "CPRT": "工业/物流",
    "LIN": "工业/物流",
    # 金融科技
    "PYPL": "金融科技", "COIN": "金融科技",
    # 新能源/公用
    "CEG": "新能源/公用", "ENPH": "新能源/公用", "EXC": "新能源/公用",
    "XEL": "新能源/公用", "BKR": "新能源/公用", "FANG": "新能源/公用",
    "AEP": "新能源/公用",
    # 新兴科技/AI
    "TSLA": "新兴科技/AI", "PLTR": "新兴科技/AI",
    "AXON": "新兴科技/AI", "SMCI": "新兴科技/AI",
}

# 板块展示顺序
SECTOR_ORDER = [
    "半导体", "大型科技", "软件/SaaS", "互联网/电商", "医疗健康",
    "消费/零售", "媒体/娱乐", "工业/物流", "金融科技", "新能源/公用", "新兴科技/AI",
]
