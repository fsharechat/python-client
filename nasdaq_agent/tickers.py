# 纳斯达克100成分股列表（截至2026年5月，含4月20日调整：+SNDK -TEAM）
NASDAQ100_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "TSLA", "AVGO", "COST",
    "NFLX", "AMD", "ASML", "LIN", "ADBE", "QCOM", "INTU", "AMAT", "BKNG", "ISRG",
    "VRTX", "TXN", "ADP", "PANW", "MU", "GILD", "LRCX", "ADI", "REGN", "KLAC",
    "SNPS", "CDNS", "MDLZ", "SBUX", "INTC", "CTAS", "ORLY", "NXPI", "MRVL", "CEG",
    "PYPL", "CRWD", "WDAY", "CSX", "ABNB", "PAYX", "DXCM", "IDXX", "ODFL", "ROST",
    "FAST", "VRSK", "PCAR", "AEP", "CHTR", "FANG", "MNST", "SNDK", "MELI", "PDD",
    "GEHC", "DDOG", "ARM", "ZS", "ROP", "KDP", "CPRT", "CSGP", "EXC", "XEL",
    "BKR", "CTSH", "MCHP", "EA", "WBD", "KHC", "TTWO", "FTNT", "APP", "PLTR",
    "AXON", "DASH", "CSCO", "CMCSA", "TMUS", "AMGN", "ALNY", "INSM", "ADSK", "HON",
    "MAR", "PEP", "WMT", "CCEP", "SHOP", "MPWR", "STX", "WDC", "MSTR", "FER", "TRI",
]

# ticker → 板块中文名
NASDAQ100_SECTOR_MAP: dict[str, str] = {
    # 半导体（21只）
    "NVDA": "半导体", "AMD": "半导体", "AVGO": "半导体", "QCOM": "半导体",
    "INTC": "半导体", "TXN": "半导体", "AMAT": "半导体", "LRCX": "半导体",
    "KLAC": "半导体", "NXPI": "半导体", "MRVL": "半导体", "MCHP": "半导体",
    "ADI": "半导体", "MU": "半导体", "ASML": "半导体", "ARM": "半导体",
    "SNPS": "半导体", "CDNS": "半导体", "MPWR": "半导体",
    "STX": "半导体", "WDC": "半导体", "SNDK": "半导体",
    # 大型科技（7只）
    "AAPL": "大型科技", "MSFT": "大型科技", "GOOGL": "大型科技",
    "GOOG": "大型科技", "META": "大型科技", "AMZN": "大型科技",
    "CSCO": "大型科技",
    # 软件/SaaS（14只）
    "ADBE": "软件/SaaS", "INTU": "软件/SaaS", "PANW": "软件/SaaS",
    "CRWD": "软件/SaaS", "WDAY": "软件/SaaS", "DDOG": "软件/SaaS",
    "ZS": "软件/SaaS", "FTNT": "软件/SaaS", "CTSH": "软件/SaaS",
    "ROP": "软件/SaaS", "CSGP": "软件/SaaS", "ADSK": "软件/SaaS",
    "TRI": "软件/SaaS",
    # 互联网/电商（7只）
    "BKNG": "互联网/电商", "ABNB": "互联网/电商", "MELI": "互联网/电商",
    "PDD": "互联网/电商", "DASH": "互联网/电商", "APP": "互联网/电商",
    "SHOP": "互联网/电商",
    # 医疗健康（10只）
    "ISRG": "医疗健康", "VRTX": "医疗健康", "GILD": "医疗健康",
    "REGN": "医疗健康", "IDXX": "医疗健康", "DXCM": "医疗健康",
    "GEHC": "医疗健康", "ALNY": "医疗健康", "AMGN": "医疗健康",
    "INSM": "医疗健康",
    # 消费/零售（12只）
    "COST": "消费/零售", "SBUX": "消费/零售", "ORLY": "消费/零售",
    "ROST": "消费/零售", "MDLZ": "消费/零售", "MNST": "消费/零售",
    "KDP": "消费/零售", "KHC": "消费/零售", "CCEP": "消费/零售",
    "MAR": "消费/零售", "PEP": "消费/零售", "WMT": "消费/零售",
    # 媒体/娱乐（7只）
    "NFLX": "媒体/娱乐", "CHTR": "媒体/娱乐", "WBD": "媒体/娱乐",
    "EA": "媒体/娱乐", "TTWO": "媒体/娱乐", "CMCSA": "媒体/娱乐",
    "TMUS": "媒体/娱乐",
    # 工业/物流（12只）
    "ADP": "工业/物流", "PAYX": "工业/物流", "PCAR": "工业/物流",
    "ODFL": "工业/物流", "CSX": "工业/物流", "FAST": "工业/物流",
    "VRSK": "工业/物流", "CTAS": "工业/物流", "CPRT": "工业/物流",
    "LIN": "工业/物流", "FER": "工业/物流", "HON": "工业/物流",
    # 金融科技（2只）
    "PYPL": "金融科技", "MSTR": "金融科技",
    # 新能源/公用（6只）
    "CEG": "新能源/公用", "EXC": "新能源/公用", "XEL": "新能源/公用",
    "BKR": "新能源/公用", "FANG": "新能源/公用", "AEP": "新能源/公用",
    # 新兴科技/AI（3只）
    "TSLA": "新兴科技/AI", "PLTR": "新兴科技/AI", "AXON": "新兴科技/AI",
}

# 板块展示顺序
SECTOR_ORDER = [
    "半导体", "大型科技", "软件/SaaS", "互联网/电商", "医疗健康",
    "消费/零售", "媒体/娱乐", "工业/物流", "金融科技", "新能源/公用", "新兴科技/AI",
]
