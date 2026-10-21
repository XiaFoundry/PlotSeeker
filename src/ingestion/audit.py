"""Streaming audit of the Bangumi Archive JSONL files.

The audit only reads ``data/raw``. Every number in the generated report comes
from an actual pass over the files; nothing is assumed from planning documents.

Design notes
------------
* Files are read line by line as bytes. The raw bytes are hashed in the same
  pass that feeds ``json.loads`` so the report can record data provenance
  without a second full read.
* Text lengths are kept in ``array("I")`` to bound memory for the 1.7M-episode
  file.
* Sampling uses a fixed-seed reservoir sampler. Full scans are used for ID
  sets, field completeness, relation integrity, and (for subject / character /
  person) infobox parsing. Expensive text-quality judgements are sampled and
  are labelled as such in the report.
* Numeric code fields are reported as distributions plus example names. Their
  meaning is deliberately not interpreted.
"""

from __future__ import annotations

import json
import os
import platform
import random
import re
import sys
import time
from array import array
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from config import ALL_DATA_FILES, DOCS_DIR, MAIN_ENTITY_FILES, RAW_DIR
from ingestion.infobox import (
    CN_NAME_KEYS,
    NAME_KEYS,
    cn_name_from_infobox,
    parse_aliases,
    parse_infobox,
)
from ingestion.jsonl import FileStats, iter_jsonl
from ingestion.resources import peak_rss_mb
from ingestion.text_quality import QUALITY_LABELS, classify_episode_description

MAX_CODE_EXAMPLES = 3
MAX_QUALITY_EXAMPLES = 2

# ---------------------------------------------------------------------------
# Small statistics helpers
# ---------------------------------------------------------------------------


class LengthStats:
    """Track counts and length percentiles without keeping Python int objects."""

    __slots__ = ("total", "empty", "_values")

    def __init__(self) -> None:
        self.total = 0
        self.empty = 0
        self._values = array("I")

    @property
    def non_empty(self) -> int:
        return self.total - self.empty

    def add(self, value: Any) -> None:
        self.total += 1
        if not isinstance(value, str) or not value.strip():
            self.empty += 1
            return
        length = len(value)
        self._values.append(min(length, 4_000_000_000))

    def summary(self) -> dict[str, Any]:
        values = sorted(self._values)
        if not values:
            return {
                "count": self.total,
                "empty": self.empty,
                "non_empty": 0,
                "min": 0,
                "p50": 0,
                "p90": 0,
                "p99": 0,
                "max": 0,
                "mean": 0.0,
            }
        total_chars = sum(values)
        return {
            "count": self.total,
            "empty": self.empty,
            "non_empty": len(values),
            "min": values[0],
            "p50": _percentile(values, 50),
            "p90": _percentile(values, 90),
            "p99": _percentile(values, 99),
            "max": values[-1],
            "mean": round(total_chars / len(values), 1),
        }


def _percentile(sorted_values: list[int], p: float) -> int:
    if not sorted_values:
        return 0
    index = int(round((len(sorted_values) - 1) * p / 100.0))
    index = max(0, min(index, len(sorted_values) - 1))
    return sorted_values[index]


class ReservoirSampler:
    def __init__(self, k: int, seed: int = 42) -> None:
        self.k = max(0, k)
        self._rng = random.Random(seed)
        self.seen = 0
        self.items: list[Any] = []

    def add(self, item: Any) -> None:
        self.seen += 1
        if self.k == 0:
            return
        if len(self.items) < self.k:
            self.items.append(item)
            return
        index = self._rng.randrange(self.seen)
        if index < self.k:
            self.items[index] = item


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, dict, tuple, set)):
        return len(value) == 0
    return False


def _year_of(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        return "缺失"
    match = re.match(r"\s*(\d{4})", value)
    if not match:
        return "无法解析"
    year = int(match.group(1))
    if 1800 <= year <= 2200:
        return str(year)
    return "无法解析"


# ---------------------------------------------------------------------------
# Language and episode-description heuristics
# ---------------------------------------------------------------------------

_SIMP_TRAD_PAIRS = """
们們 个個 这這 来來 时時 说說 会會 对對 后後 里裡 发發 还還 学學 国國 过過
动動 开開 关關 机機 体體 点點 电電 头頭 长長 门門 东東 车車 马馬 语語 读讀
写寫 无無 与與 为為 么麼 样樣 岁歲 钱錢 吗嗎 听聽 见見 让讓 给給 从從 爱愛
戏戲 剧劇 尔爾 战戰 争爭 声聲 优優 游遊 乐樂 团團 场場 实實 现現 际際 应應
该該 专專 业業 产產 华華 龙龍 凤鳳 飞飛 鸟鳥 鱼魚 几幾 张張 阳陽 阴陰 云雲
风風 网網 络絡 软軟 资資 讯訊 页頁 连連 档檔 设設 选選 单單 预預 确確 认認
删刪 储儲 启啟 闭閉 错錯 误誤 库庫 寻尋 结結 显顯 图圖 画畫 话話 节節 员員
队隊 军軍 师師 园園 区區 乡鄉 镇鎮 县縣 岛島 屿嶼 湾灣 边邊 线線 丝絲 红紅
绿綠 蓝藍 黄黃 银銀 铁鐵 钢鋼 脑腦 键鍵 盘盤 萤螢 统統 夹夾 记記 忆憶 处處
温溫 传傳 术術 艺藝 万萬 临臨 义義 乌烏 乔喬 习習 书書 买買 乱亂 亏虧 亚亞
亲親 亿億 仅僅 仓倉 仪儀 价價 众眾 伟偉 伤傷 伦倫 伪偽 佣傭 侠俠 侣侶 侦偵
侧側 侨僑 债債 倾傾 偿償 儿兒 兰蘭 兴興 养養 兽獸 内內 冈岡 册冊 农農 冯馮
冲衝 决決 况況 冻凍 净淨 凉涼 减減 凑湊 凛凜 凭憑 凯凱 击擊 刘劉 则則 刚剛
创創 别別 剂劑 办辦 务務 励勵 劲勁 劳勞 势勢 勋勳 医醫 协協 卖賣 卢盧 卫衛
却卻 厂廠 厅廳 历歷 厉厲 压壓 厌厭 厨廚 县縣 变變 叶葉 号號 叹嘆 吓嚇 吕呂
吨噸 吴吳 呐吶 呛嗆 呜嗚 咏詠 响響 哑啞 唤喚 啸嘯 喷噴 嘱囑 围圍 圆圓 圣聖
坏壞 块塊 坚堅 坝壩 坟墳 垄壟 垒壘 垦墾 垫墊 墙牆 壮壯 壳殼 壶壺 备備 复復
够夠 夸誇 夺奪 奋奮 奖獎 妆妝 妇婦 妈媽 娱娛 婴嬰 孙孫 孪孿 宁寧 宝寶 宠寵
审審 宪憲 宫宮 宽寬 宾賓 寝寢 导導 寿壽 将將 尘塵 尝嘗 尧堯 尴尷 尸屍 尽盡
层層 屉屜 届屆 属屬 岂豈 岗崗 岭嶺 岳嶽 峡峽 峥崢 峦巒 崭嶄 嵘嶸 巅巔 巩鞏
币幣 帅帥 帏幃 帐帳 帘簾 帜幟 带帶 帮幫 帼幗 幂冪 并並 广廣 庄莊 庆慶 庐廬
库庫 庙廟 庞龐 废廢 异異 弃棄 弥彌 弯彎 弹彈 强強 归歸 当當 录錄 彦彥 彻徹
径徑 徕徠 忏懺 忧憂 怀懷 态態 怂慫 怃憮 怄慪 怅悵 怆愴 怜憐 总總 怼懟 恋戀
恒恆 恳懇 恶惡 恸慟 恹懨 恺愷 恻惻 恼惱 恽惲 悦悅 悬懸 悭慳 悯憫 惊驚 惧懼
惨慘 惩懲 惫憊 惬愜 惭慚 惮憚 惯慣 愠慍 愤憤 愦憒 愿願 慑懾 懑懣 懒懶 戆戇
戋戔 戗戧 戬戩 户戶 扑撲 执執 扩擴 扪捫 扫掃 扬揚 扰擾 抚撫 抛拋 抠摳 抡掄
抢搶 护護 报報 担擔 拟擬 拢攏 拣揀 拥擁 拦攔 拧擰 拨撥 择擇 挂掛 挚摯 挛攣
挝撾 挞撻 挟挾 挠撓 挡擋 挣掙 挤擠 挥揮 捞撈 损損 捡撿 换換 捣搗 据據 掳擄
掴摑 掷擲 掸撣 掺摻 掼摜 揽攬 搀攙 搁擱 搂摟 搅攪 携攜 摄攝 摆擺 摇搖 摈擯
摊攤 撑撐 撵攆 撷擷 撸擼 撺攛 擞擻 攒攢 敌敵 敛斂 数數 斋齋 斗鬥 斩斬 断斷
旧舊 旷曠 昙曇 昼晝 晋晉 晒曬 晓曉 晕暈 晖暉 暂暫 暧曖 杀殺 杂雜 权權 条條
杨楊 杰傑 极極 构構 枢樞 枣棗 枪槍 枫楓 枭梟 柜櫃 柠檸 标標 栈棧 栋棟 栌櫨
栎櫟 栏欄 树樹 栖棲 栾欒 桡橈 桢楨 桤榿 桥橋 桦樺 桧檜 桨槳 桩樁 梦夢 检檢
椟櫝 椭橢 楼樓 榄欖 榇櫬 榈櫚 榉櫸 槛檻 槟檳 横橫 樯檣 樱櫻 橱櫥 橹櫓 橼櫞
欢歡 欧歐 歼殲 殇殤 残殘 殒殞 殓殮 殚殫 殡殯 殴毆 毁毀 毂轂 毕畢 毙斃 毡氈
气氣 氢氫 氩氬 汇匯 汉漢 汤湯 汹洶 沟溝 没沒 沣灃 沤漚 沥瀝 沦淪 沧滄 沪滬
泞濘 泪淚 泷瀧 泸瀘 泺濼 泻瀉 泼潑 泽澤 泾涇 洁潔 洒灑 洼窪 浃浹 浅淺 浆漿
浇澆 浊濁 测測 济濟 浏瀏 浑渾 浒滸 浓濃 浔潯 涂塗 涌湧 涛濤 涝澇 涞淶 涟漣
涡渦 涣渙 涤滌 润潤 涧澗 涨漲 涩澀 淀澱 渊淵 渍漬 渎瀆 渐漸 渔漁 渗滲 湿濕
溃潰 溅濺 满滿 滚滾 滞滯 滢瀅 滤濾 滥濫 滦灤 滨濱 滩灘 潇瀟 潋瀲 潍濰 潜潛
澜瀾 濑瀨 濒瀕 灏灝 灭滅 灯燈 灵靈 灾災 灿燦 炀煬 炉爐 炖燉 炜煒 炝熗 炼煉
炽熾 烁爍 烂爛 烛燭 烟煙 烦煩 烧燒 烨燁 烩燴 烫燙 烬燼 热熱 焕煥 焖燜 焘燾
爱愛 爷爺 牍牘 牵牽 牺犧 犊犢 状狀 犷獷 犸獁 犹猶 狈狽 狞獰 独獨 狭狹 狮獅
狯獪 狰猙 狱獄 狲猻 猎獵 猕獼 猪豬 猫貓 献獻 獭獺 玑璣 玛瑪 玮瑋 环環 玺璽
珐琺 珑瓏 珰璫 珲琿 琏璉 琐瑣 琼瓊 瑶瑤 瑷璦 璎瓔 瓒瓚 瓯甌 畅暢 畴疇 疖癤
疗療 疟瘧 疠癘 疡瘍 疮瘡 疯瘋 痈癰 痉痙 痒癢 痨癆 痪瘓 痫癇 瘅癉 瘘瘻 瘪癟
瘫癱 瘾癮 瘿癭 癞癩 癣癬 癫癲 皑皚 皱皺 皲皸 盏盞 盐鹽 监監 盖蓋 盗盜 眍瞘
眦眥 睁睜 睐睞 睑瞼 瞒瞞 瞩矚 矫矯 矶磯 矾礬 矿礦 砀碭 码碼 砖磚 砚硯 砺礪
砻礱 砾礫 础礎 硕碩 硖硤 硗磽 确確 碍礙 碛磧 碜磣 礼禮 祎禕 祢禰 祯禎 祷禱
祸禍 禀稟 禄祿 禅禪 离離 秃禿 秆稈 种種 积積 称稱 秽穢 税稅 稣穌 稳穩 穑穡
穷窮 窃竊 窍竅 窑窯 窜竄 窝窩 窥窺 窦竇 竖豎 竞競 笃篤 笋筍 笔筆 笺箋 笼籠
筑築 筛篩 筝箏 筹籌 签簽 简簡 箦簀 箧篋 箩籮 箪簞 箫簫 篑簣 篓簍 篮籃 篱籬
籁籟 类類 籼秈 粝糲 粤粵 粪糞 粮糧 紧緊 纠糾 纡紆 纣紂 纤纖 纥紇 约約 级級
纨紈 纪紀 纫紉 纬緯 纭紜 纯純 纰紕 纱紗 纲綱 纳納 纵縱 纶綸 纷紛 纸紙 纹紋
纺紡 纽紐 纾紓 绀紺 练練 组組 绅紳 细細 织織 终終 绉縐 绊絆 绌絀 绍紹 绎繹
经經 绑綁 绒絨 绕繞 绘繪 绚絢 绛絳 绝絕 绞絞 绠綆 绡綃 绢絹 绣繡 绥綏 绦絛
继繼 绨綈 绩績 绪緒 绫綾 续續 绮綺 绯緋 绰綽 绳繩 维維 绵綿 绶綬 绷繃 绸綢
绺綹 绻綣 综綜 绽綻 绾綰 缀綴 缁緇 缂緙 缃緗 缄緘 缅緬 缆纜 缇緹 缉緝 缎緞
缓緩 缔締 缕縷 编編 缘緣 缙縉 缚縛 缛縟 缜縝 缝縫 缟縞 缠纏 缢縊 缤繽 缥縹
缦縵 缨纓 缩縮 缪繆 缫繅 缬纈 缭繚 缮繕 缯繒 缰韁 缴繳 罂罌 罗羅 罚罰 罢罷
罴羆 羁羈 翘翹 耸聳 耻恥 聂聶 聋聾 职職 联聯 聪聰 肃肅 肠腸 肤膚 肮骯 肾腎
肿腫 胀脹 胁脅 胆膽 胜勝 胧朧 胪臚 胫脛 胶膠 脉脈 脍膾 脏臟 脐臍 脓膿 脚腳
脱脫 脸臉 腊臘 腻膩 腾騰 舆輿 舰艦 舱艙 艰艱 艳艷 芈羋 芗薌 芜蕪 芦蘆 苇葦
苋莧 苍蒼 苎苧 苏蘇 苹蘋 茎莖 茏蘢 茑蔦 茔塋 茕煢 茧繭 荆荊 荐薦 荚莢 荛蕘
荜蓽 荞蕎 荟薈 荠薺 荡蕩 荣榮 荤葷 荧熒 荨蕁 荩藎 荪蓀 荫蔭 药藥 莅蒞 莱萊
莲蓮 莳蒔 莴萵 获獲 莹瑩 莺鶯 莼蓴 萝蘿 营營 萦縈 萧蕭 萨薩 葱蔥 蒋蔣 蒌蔞
蓟薊 蓠蘺 蓣蕷 蓥鎣 蓦驀 蔷薔 蔹蘞 蔺藺 蕲蘄 薮藪 藓蘚 虏虜 虑慮 虚虛 虫蟲
虬虯 虽雖 虾蝦 蚀蝕 蚁蟻 蚂螞 蚕蠶 蚬蜆 蛊蠱 蛎蠣 蛏蟶 蛮蠻 蛰蟄 蛱蛺 蛲蟯
蛳螄 蛴蠐 蜕蛻 蜗蝸 蝇蠅 蝈蟈 蝉蟬 蝼螻 蝾蠑 螨蟎 衅釁 衔銜 补補 衬襯 衮袞
袄襖 袅裊 袜襪 袭襲 装裝 裆襠 裢褳 裤褲 褛褸 褴襤 观觀 规規 觅覓 视視 览覽
觉覺 觊覬 觋覡 觌覿 觎覦 觏覯 觐覲 觑覷 觞觴 触觸 誉譽 誊謄 计計 订訂 讣訃
讥譏 讦訐 讧訌 讨討 讪訕 讫訖 训訓 议議 讯訊 记記 讲講 讳諱 讴謳 讵詎 讶訝
讷訥 许許 讹訛 论論 讼訟 讽諷 访訪 诀訣 证證 评評 诅詛 识識 诈詐 诉訴 诊診
诋詆 诌謅 词詞 诏詔 译譯 诒詒 诓誆 诔誄 试試 诖詿 诗詩 诘詰 诙詼 诚誠 诛誅
诜詵 诞誕 诟詬 诠詮 诡詭 询詢 诣詣 诤諍 详詳 诧詫 诨諢 诩詡 诫誡 诬誣 诮誚
诰誥 诱誘 诲誨 诳誑 诵誦 请請 诸諸 诺諾 诼諑 诽誹 课課 诿諉 谀諛 谁誰 谂諗
调調 谄諂 谅諒 谆諄 谈談 谊誼 谋謀 谌諶 谍諜 谎謊 谏諫 谐諧 谑謔 谒謁 谓謂
谔諤 谕諭 谖諼 谗讒 谙諳 谚諺 谛諦 谜謎 谝諞 谟謨 谠讜 谢謝 谣謠 谤謗 谦謙
谧謐 谨謹 谩謾 谪謫 谬謬 谭譚 谱譜 谲譎 谴譴 谶讖 贝貝 贞貞 负負 贡貢 财財
责責 贤賢 败敗 账賬 货貨 质質 贩販 贪貪 贫貧 贬貶 购購 贮貯 贯貫 贰貳 贱賤
贲賁 贴貼 贵貴 贷貸 贸貿 费費 贺賀 贻貽 贼賊 贾賈 贿賄 赁賃 赂賂 赃贓 资資
赅賅 赈賑 赊賒 赋賦 赌賭 赎贖 赏賞 赐賜 赓賡 赔賠 赖賴 赘贅 赚賺 赛賽 赞贊
赠贈 赡贍 赢贏 赣贛 赵趙 赶趕 趋趨 车車 轧軋 轨軌 轩軒 转轉 轭軛 轮輪 软軟
轰轟 轱軲 轴軸 轶軼 轸軫 轻輕 轼軾 载載 轿轎 较較 辅輔 辆輛 辈輩 辉輝 辊輥
辍輟 辐輻 辑輯 输輸 辕轅 辖轄 辗輾 辘轆 辙轍 辞辭 辩辯 辫辮 辽遼 达達 迁遷
迈邁 运運 进進 远遠 违違 迟遲 适適 逊遜 递遞 逻邏 遗遺 遥遙 邓鄧 邝鄺 邬鄔
邮郵 邹鄒 邻鄰 郏郟 郑鄭 郓鄆 郦酈 郧鄖 郸鄲 酝醞 酱醬 酿釀 释釋 鉴鑑 銮鑾
针針 钉釘 钊釗 钋釙 钌釕 钍釷 钎釺 钏釧 钐釤 钒釩 钓釣 钕釹 钗釵 钙鈣 钚鈈
钛鈦 钝鈍 钞鈔 钟鐘 钠鈉 钡鋇 钣鈑 钤鈐 钥鑰 钦欽 钧鈞 钨鎢 钩鉤 钪鈧 钫鈁
钬鈥 钮鈕 钯鈀 钰鈺 钲鉦 钳鉗 钴鈷 钵缽 钶鈳 钷鉕 钹鈸 钺鉞 钻鑽 钼鉬 钽鉭
钾鉀 钿鈿 铀鈾 铂鉑 铃鈴 铄鑠 铅鉛 铆鉚 铈鈰 铉鉉 铊鉈 铋鉍 铌鈮 铍鈹 铎鐸
铐銬 铑銠 铒鉺 铕銪 铖鋮 铗鋏 铙鐃 铛鐺 铜銅 铝鋁 铟銦 铠鎧 铡鍘 铢銖 铣銑
铤鋌 铥銩 铧鏵 铨銓 铩鎩 铬鉻 铭銘 铮錚 铯銫 铰鉸 铱銥 铲鏟 铳銃 铵銨 铷銣
铸鑄 铺鋪 铼錸 链鏈 铿鏗 销銷 锁鎖 锂鋰 锃鋥 锄鋤 锅鍋 锆鋯 锇鋨 锈銹 锉銼
锋鋒 锌鋅 锍鋶 锎鐦 锏鐧 锐銳 锑銻 锒鋃 锔鋦 锕錒 锗鍺 锘鍩 锚錨 锛錛 锝鍀
锞錁 锟錕 锡錫 锢錮 锣鑼 锤錘 锥錐 锦錦 锨鍁 锭錠 锯鋸 锰錳 锱錙 锲鍥 锴鍇
锵鏘 锶鍶 锷鍔 锸鍤 锹鍬 锻鍛 镀鍍 镁鎂 镂鏤 镇鎮 镉鎘 镊鑷 镌鐫 镍鎳 镏鎦
镐鎬 镑鎊 镒鎰 镓鎵 镔鑌 镖鏢 镗鏜 镘鏝 镜鏡 镝鏑 镞鏃 镟鏇 镣鐐 镤鏷 镥鑥
镦鐓 镧鑭 镨鐠 镩鑹 镪鏹 镫鐙 镬鑊 镭鐳 镯鐲 镰鐮 镱鐿 镲鑔 镳鑣 镶鑲 长長
闩閂 闪閃 闫閆 问問 闯闖 闰閏 闲閒 间間 闵閔 闷悶 闸閘 闹鬧 闺閨 闻聞 闽閩
闾閭 阀閥 阁閣 阂閡 阃閫 阅閱 阆閬 阈閾 阉閹 阎閻 阐闡 阑闌 阔闊 阕闋 阖闔
阗闐 阙闕 队隊 阳陽 阴陰 阵陣 阶階 际際 陆陸 陇隴 陈陳 陕陝 陨隕 险險 随隨
隐隱 隶隸 难難 雏雛 雳靂 雾霧 霁霽 霉黴 靥靨 静靜 韦韋 韧韌 韩韓 韪韙 韫韞
韬韜 韵韻 顶頂 顷頃 项項 顺順 须須 顽頑 顾顧 顿頓 颁頒 颂頌 预預 颅顱 领領
颇頗 颈頸 颊頰 颌頜 颐頤 频頻 颓頹 颔頷 颖穎 颗顆 题題 颚顎 颛顓 颜顏 额額
颞顳 颟顢 颠顛 颤顫 颦顰 颧顴 飏颺 飒颯 飓颶 飕颼 飘飄 飙飆 飨饗 饥飢 饧餳
饨飩 饪飪 饬飭 饭飯 饮飲 饯餞 饰飾 饱飽 饲飼 饴飴 饵餌 饶饒 饷餉 饺餃 饼餅
饿餓 馁餒 馄餛 馅餡 馆館 馈饋 馊餿 馋饞 馍饃 馏餾 馐饈 馑饉 馒饅 馔饌 馕饢
驭馭 驮馱 驯馴 驰馳 驱驅 驳駁 驴驢 驶駛 驷駟 驹駒 驻駐 驼駝 驾駕 驿驛 骂罵
骄驕 骆駱 骇駭 骈駢 骊驪 骋騁 验驗 骏駿 骑騎 骗騙 骚騷 骡騾 骤驟 骥驥 髅髏
髋髖 髌髕 鬓鬢 魇魘 魉魎 鲁魯 鲅鮁 鲆鮃 鲇鮎 鲈鱸 鲍鮑 鲎鱟 鲑鮭 鲔鮪 鲛鮫
鲜鮮 鲟鱘 鲤鯉 鲨鯊 鲫鯽 鲭鯖 鲮鯪 鲱鯡 鲲鯤 鲳鯧 鲵鯢 鲶鯰 鲷鯛 鲸鯨 鲻鯔
鲽鰈 鳄鱷 鳅鰍 鳆鰒 鳇鰉 鳊鯿 鳌鰲 鳍鰭 鳎鰨 鳏鰥 鳐鰩 鳕鱈 鳖鱉 鳗鰻 鳙鱅
鳜鱖 鳝鱔 鳞鱗 鳟鱒 鸠鳩 鸡雞 鸢鳶 鸣鳴 鸥鷗 鸦鴉 鸨鴇 鸪鴣 鸫鶇 鸬鸕 鸭鴨
鸯鴦 鸱鴟 鸳鴛 鸵鴕 鸶鷥 鸷鷙 鸸鴯 鸽鴿 鸾鸞 鸿鴻 鹂鸝 鹃鵑 鹄鵠 鹅鵝 鹈鵜
鹉鵡 鹊鵲 鹌鵪 鹏鵬 鹑鶉 鹕鶘 鹗鶚 鹘鶻 鹚鷀 鹜鶩 鹞鷂 鹣鶼 鹤鶴 鹦鸚 鹧鷓
鹩鷯 鹪鷦 鹭鷺 鹰鷹 鹳鸛 麦麥 麸麩 黄黃 黉黌 黩黷 黾黽 鼋黿 鼍鼉 鼹鼴 齐齊
齑齏 齿齒 龀齔 龃齟 龄齡 龅齙 龆齠 龇齜 龈齦 龉齬 龊齪 龋齲 龌齷 龚龔 龛龕
龟龜
"""

SIMP_TO_TRAD = {pair[0]: pair[1] for pair in _SIMP_TRAD_PAIRS.split() if len(pair) == 2}
SIMP_CHARS = "".join(SIMP_TO_TRAD.keys())
TRAD_CHARS = "".join(dict.fromkeys(SIMP_TO_TRAD.values()))
assert not (set(SIMP_CHARS) & set(TRAD_CHARS)), "simplified/traditional character sets overlap"

HAN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
KANA_RE = re.compile(r"[\u3040-\u30ff\u31f0-\u31ff]")
LATIN_RE = re.compile(r"[A-Za-z\u00c0-\u024f]")
SIMP_RE = re.compile("[" + re.escape(SIMP_CHARS) + "]")
TRAD_RE = re.compile("[" + re.escape(TRAD_CHARS) + "]")

LANG_LABELS = {
    "empty": "空文本",
    "japanese": "日文（含假名）",
    "chinese_simplified": "简体中文（特征近似）",
    "chinese_traditional": "繁体中文（特征近似）",
    "chinese_mixed": "简繁混排",
    "han_unclassified": "汉字（简繁特征不足）",
    "latin": "无汉字/假名，含拉丁字母",
    "other": "其他",
}


def classify_text(text: Any) -> str:
    """Rough language/script classification for coverage statistics.

    Traditional/simplified separation is a heuristic based on a curated set of
    exclusive character pairs. It is intentionally reported as approximate.
    """

    if not isinstance(text, str) or not text.strip():
        return "empty"
    if KANA_RE.search(text):
        return "japanese"
    if HAN_RE.search(text):
        simplified_hits = len(SIMP_RE.findall(text))
        traditional_hits = len(TRAD_RE.findall(text))
        if simplified_hits and traditional_hits:
            total = simplified_hits + traditional_hits
            if traditional_hits / total >= 0.8:
                return "chinese_traditional"
            if traditional_hits / total <= 0.2:
                return "chinese_simplified"
            return "chinese_mixed"
        if traditional_hits:
            return "chinese_traditional"
        if simplified_hits:
            return "chinese_simplified"
        return "han_unclassified"
    if LATIN_RE.search(text):
        return "latin"
    return "other"


# ---------------------------------------------------------------------------
# Entity / relation audit containers
# ---------------------------------------------------------------------------


@dataclass
class InfoboxAudit:
    entity: str
    total: int = 0
    non_empty: int = 0
    parsed: int = 0
    failed: int = 0
    keys: Counter = field(default_factory=Counter)
    name_key_docs: Counter = field(default_factory=Counter)
    alias_docs: int = 0
    alias_total: int = 0
    alias_hist: Counter = field(default_factory=Counter)
    multiline_alias_docs: int = 0
    extra_cn_name_docs: int = 0
    alias_examples: list[tuple[int, list[str]]] = field(default_factory=list)
    raw_example: dict[str, Any] | None = None
    failure_examples: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class EntityAudit:
    entity: str
    file: FileStats
    ids: set[int] = field(default_factory=set)
    duplicate_id_count: int = 0
    duplicate_id_examples: list[int] = field(default_factory=list)
    invalid_id_count: int = 0
    missing: dict[str, int] = field(default_factory=dict)
    lengths: dict[str, LengthStats] = field(default_factory=dict)
    counters: dict[str, Counter] = field(default_factory=Counter)
    code_examples: dict[str, dict[Any, list[str]]] = field(default_factory=dict)
    language: Counter = field(default_factory=Counter)
    language_total: int = 0
    infobox: InfoboxAudit | None = None
    quality: Counter = field(default_factory=Counter)
    quality_examples: dict[str, list[str]] = field(default_factory=dict)
    quality_sample_size: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class RelationAudit:
    name: str
    file: FileStats
    natural_key: tuple[str, ...]
    duplicate_count: int = 0
    duplicate_examples: list[dict[str, Any]] = field(default_factory=list)
    counters: dict[str, Counter] = field(default_factory=Counter)
    empty_counts: dict[str, int] = field(default_factory=dict)
    orphans: dict[str, dict[str, Any]] = field(default_factory=dict)
    coverage: dict[str, set[int]] = field(default_factory=dict)
    typed_membership: Counter = field(default_factory=Counter)


ENTITY_SPECS: dict[str, dict[str, Any]] = {
    "subject": {
        "fields": [
            "id",
            "type",
            "name",
            "name_cn",
            "infobox",
            "platform",
            "summary",
            "nsfw",
            "tags",
            "meta_tags",
            "score",
            "rank",
            "date",
            "favorite",
            "series",
        ],
        "lengths": ["name", "name_cn", "summary"],
        "counter_examples": ["type", "platform"],
    },
    "character": {
        "fields": ["id", "role", "name", "infobox", "summary", "comments", "collects"],
        "lengths": ["name", "summary"],
        "counter_examples": ["role"],
    },
    "person": {
        "fields": ["id", "name", "type", "career", "infobox", "summary", "comments", "collects"],
        "lengths": ["name", "summary"],
        "counter_examples": ["type"],
    },
    "episode": {
        "fields": [
            "id",
            "name",
            "name_cn",
            "description",
            "airdate",
            "disc",
            "duration",
            "subject_id",
            "sort",
            "type",
        ],
        "lengths": ["name", "name_cn", "description"],
        "counter_examples": ["type"],
    },
}

RELATION_SPECS: list[dict[str, Any]] = [
    {
        "name": "subject-characters",
        "filename": "subject-characters.jsonlines",
        "natural_key": ("subject_id", "character_id"),
        "counters": ["type"],
        "empty_fields": [],
        "endpoints": [("subject_id", "subject"), ("character_id", "character")],
        "coverage": {"subject_id": "subjects", "character_id": "characters"},
    },
    {
        "name": "subject-persons",
        "filename": "subject-persons.jsonlines",
        "natural_key": ("subject_id", "person_id", "position"),
        "counters": ["position"],
        "empty_fields": ["appear_eps"],
        "endpoints": [("subject_id", "subject"), ("person_id", "person")],
        "coverage": {"subject_id": "subjects", "person_id": "persons"},
    },
    {
        "name": "subject-relations",
        "filename": "subject-relations.jsonlines",
        "natural_key": ("subject_id", "related_subject_id", "relation_type"),
        "counters": ["relation_type"],
        "empty_fields": [],
        "endpoints": [
            ("subject_id", "subject"),
            ("related_subject_id", "subject"),
        ],
        "coverage": {
            "subject_id": "subjects",
            "related_subject_id": "subjects",
        },
    },
    {
        "name": "person-characters",
        "filename": "person-characters.jsonlines",
        "natural_key": ("person_id", "subject_id", "character_id"),
        "counters": ["type"],
        "empty_fields": ["summary"],
        "endpoints": [
            ("person_id", "person"),
            ("subject_id", "subject"),
            ("character_id", "character"),
        ],
        "coverage": {
            "person_id": "persons",
            "subject_id": "subjects",
            "character_id": "characters",
        },
    },
    {
        "name": "person-relations",
        "filename": "person-relations.jsonlines",
        "natural_key": ("person_type", "person_id", "related_person_id", "relation_type"),
        "counters": ["person_type", "relation_type", "spoiler", "ended"],
        "empty_fields": [],
        "endpoints": [],
        "coverage": {},
    },
]


# ---------------------------------------------------------------------------
# Entity auditing
# ---------------------------------------------------------------------------


def _bump(
    counter: Counter,
    value: Any,
    example: str | None,
    examples: dict[Any, list[str]] | None = None,
) -> None:
    if value is None:
        return
    counter[value] += 1
    if examples is not None and example:
        bucket = examples.setdefault(value, [])
        if len(bucket) < MAX_CODE_EXAMPLES:
            bucket.append(example)


def audit_entity(
    entity: str,
    path: Path,
    *,
    limit: int | None = None,
    sample_size: int = 3000,
    seed: int = 42,
    foreign_ids: dict[str, set[int]] | None = None,
    verbose: bool = True,
) -> EntityAudit:
    spec = ENTITY_SPECS[entity]
    file_stats = FileStats(name=path.name, path=path)
    audit = EntityAudit(entity=entity, file=file_stats)
    audit.missing = {field_name: 0 for field_name in spec["fields"]}
    audit.lengths = {field_name: LengthStats() for field_name in spec["lengths"]}
    audit.counters = {name: Counter() for name in spec["counter_examples"]}
    audit.code_examples = {name: {} for name in spec["counter_examples"]}
    if entity == "subject":
        for name in ("nsfw", "date_year", "tags", "meta_tags"):
            audit.counters[name] = Counter()
    elif entity == "person":
        for name in ("career", "career_len"):
            audit.counters[name] = Counter()
    elif entity == "episode":
        for name in ("duration", "airdate_year"):
            audit.counters[name] = Counter()
    audit.extra = {"summary_ids": set()}
    if entity == "episode":
        audit.extra["description_ids"] = set()
        audit.extra["linked_ids"] = set()
        audit.extra["episode_subject_ids"] = set()
        audit.extra["orphan_subject_episodes"] = 0
        audit.extra["missing_subject_id"] = 0
    if entity == "subject":
        audit.extra["summary_safe_ids"] = set()

    if entity in ("subject", "character", "person"):
        audit.infobox = InfoboxAudit(entity=entity)

    episode_sampler = ReservoirSampler(sample_size, seed) if entity == "episode" else None
    subject_ids = (foreign_ids or {}).get("subject", set())
    started = time.perf_counter()

    for _, record in iter_jsonl(path, file_stats, limit=limit):
        record_id = record.get("id")
        if isinstance(record_id, int) and not isinstance(record_id, bool):
            if record_id in audit.ids:
                audit.duplicate_id_count += 1
                if len(audit.duplicate_id_examples) < 10:
                    audit.duplicate_id_examples.append(record_id)
            else:
                audit.ids.add(record_id)
        else:
            audit.invalid_id_count += 1
            record_id = None

        for field_name in spec["fields"]:
            if _is_missing(record.get(field_name)):
                audit.missing[field_name] += 1

        for field_name in spec["lengths"]:
            audit.lengths[field_name].add(record.get(field_name))

        display_name = record.get("name") or record.get("name_cn") or ""

        if entity == "subject":
            _bump(audit.counters["type"], record.get("type"), display_name, audit.code_examples["type"])
            _bump(
                audit.counters["platform"],
                record.get("platform"),
                display_name,
                audit.code_examples["platform"],
            )
            audit.counters["nsfw"][bool(record.get("nsfw"))] += 1
            audit.counters["date_year"][_year_of(record.get("date"))] += 1

            tags = record.get("tags")
            if isinstance(tags, list):
                for tag in tags:
                    if isinstance(tag, dict):
                        name = tag.get("name")
                    else:
                        name = tag
                    if isinstance(name, str) and name.strip():
                        audit.counters["tags"][name.strip()] += 1
            meta_tags = record.get("meta_tags")
            if isinstance(meta_tags, list):
                for tag in meta_tags:
                    if isinstance(tag, dict):
                        name = tag.get("name")
                    else:
                        name = tag
                    if isinstance(name, str) and name.strip():
                        audit.counters["meta_tags"][name.strip()] += 1

            if isinstance(record.get("summary"), str) and record["summary"].strip():
                if record_id is not None:
                    audit.extra["summary_ids"].add(record_id)
                    if record.get("nsfw") is False:
                        audit.extra["summary_safe_ids"].add(record_id)

        elif entity == "character":
            _bump(audit.counters["role"], record.get("role"), display_name, audit.code_examples["role"])
            if isinstance(record.get("summary"), str) and record["summary"].strip() and record_id is not None:
                audit.extra["summary_ids"].add(record_id)

        elif entity == "person":
            _bump(audit.counters["type"], record.get("type"), display_name, audit.code_examples["type"])
            careers = record.get("career")
            if isinstance(careers, list):
                audit.counters["career_len"][min(len(careers), 10)] += 1
                for career in careers:
                    if isinstance(career, str) and career.strip():
                        audit.counters["career"][career.strip()] += 1

        elif entity == "episode":
            _bump(audit.counters["type"], record.get("type"), display_name, audit.code_examples["type"])
            duration = record.get("duration")
            if isinstance(duration, str) and duration.strip():
                audit.counters["duration"][duration.strip()] += 1
            audit.counters["airdate_year"][_year_of(record.get("airdate"))] += 1

            subject_id = record.get("subject_id")
            if isinstance(subject_id, int) and not isinstance(subject_id, bool):
                if limit is None:
                    if subject_id in subject_ids:
                        audit.extra["episode_subject_ids"].add(subject_id)
                        if record_id is not None:
                            audit.extra["linked_ids"].add(record_id)
                    else:
                        audit.extra["orphan_subject_episodes"] += 1
                elif record_id is not None:
                    audit.extra["linked_ids"].add(record_id)
            else:
                audit.extra["missing_subject_id"] += 1

            description = record.get("description")
            if isinstance(description, str) and description.strip() and record_id is not None:
                audit.extra["description_ids"].add(record_id)
            if episode_sampler is not None:
                episode_sampler.add((record_id, description))

        summary = record.get("summary")
        if entity in ("subject", "character", "person") and isinstance(summary, str) and summary.strip():
            audit.language[classify_text(summary)] += 1
            audit.language_total += 1

        if audit.infobox is not None:
            _update_infobox(
                audit.infobox,
                record,
                record_id,
                entity,
            )

    file_stats.elapsed_sec = time.perf_counter() - started

    if episode_sampler is not None:
        audit.quality_sample_size = len(episode_sampler.items)
        for record_id, description in episode_sampler.items:
            category = classify_episode_description(description)
            audit.quality[category] += 1
            if isinstance(description, str) and description.strip():
                audit.language[classify_text(description)] += 1
                audit.language_total += 1
            bucket = audit.quality_examples.setdefault(category, [])
            if len(bucket) < MAX_QUALITY_EXAMPLES and isinstance(description, str) and description.strip():
                snippet = re.sub(r"\s+", " ", description).strip()
                bucket.append(f"#{record_id}: {snippet[:220]}")
        audit.extra["quality_sample_total"] = episode_sampler.seen

    if verbose:
        print(
            f"  {entity:<10} {file_stats.valid_records:>9,} records  "
            f"{file_stats.size_bytes / 1024 / 1024:>8.1f} MiB  "
            f"{file_stats.elapsed_sec:>6.1f}s",
            file=sys.stderr,
            flush=True,
        )

    return audit


def _update_infobox(
    infobox: InfoboxAudit,
    record: dict[str, Any],
    record_id: int | None,
    entity: str,
) -> None:
    infobox.total += 1
    raw = record.get("infobox")
    if not isinstance(raw, str) or not raw.strip():
        return
    infobox.non_empty += 1
    parsed = parse_infobox(raw)
    if parsed:
        infobox.parsed += 1
    else:
        infobox.failed += 1
        if len(infobox.failure_examples) < 3:
            infobox.failure_examples.append(
                {"id": record_id, "raw": raw[:400].replace("\r\n", "\n").replace("\r", "\n")}
            )
        return

    for key in parsed:
        infobox.keys[key] += 1
    for key in NAME_KEYS:
        if key in parsed:
            infobox.name_key_docs[key] += 1

    alias_raw = parsed.get("别名") or parsed.get("別名") or ""
    aliases = parse_aliases(alias_raw)
    if aliases:
        infobox.alias_docs += 1
        infobox.alias_total += len(aliases)
        infobox.alias_hist[min(len(aliases), 20)] += 1
        if "{" in alias_raw:
            infobox.multiline_alias_docs += 1
        if len(infobox.alias_examples) < 3:
            infobox.alias_examples.append((record_id or 0, aliases[:12]))

    if entity == "subject":
        cn_name = cn_name_from_infobox(parsed)
        if cn_name:
            infobox.extra_cn_name_docs += 1
            name_cn = record.get("name_cn")
            if not (isinstance(name_cn, str) and name_cn.strip()):
                infobox.name_key_docs["__name_cn_missing_but_infobox_cn__"] += 1

    if infobox.raw_example is None or (
        aliases and not infobox.raw_example["aliases"]
    ):
        infobox.raw_example = {
            "id": record_id,
            "raw": raw[:700].replace("\r\n", "\n").replace("\r", "\n"),
            "keys": list(parsed.keys())[:30],
            "aliases": aliases[:10],
        }


# ---------------------------------------------------------------------------
# Relation auditing
# ---------------------------------------------------------------------------


def audit_relation(
    spec: dict[str, Any],
    path: Path,
    *,
    ids_by_entity: dict[str, set[int]],
    limit: int | None = None,
    verbose: bool = True,
) -> RelationAudit:
    file_stats = FileStats(name=path.name, path=path)
    audit = RelationAudit(name=spec["name"], file=file_stats, natural_key=spec["natural_key"])
    audit.counters = {name: Counter() for name in spec["counters"]}
    if spec["name"] == "subject-relations":
        audit.counters["self_loop"] = Counter()
    audit.empty_counts = {name: 0 for name in spec["empty_fields"]}
    audit.orphans = {
        field_name: {"entity": entity, "count": 0, "examples": []}
        for field_name, entity in spec["endpoints"]
    }
    for target in set(spec["coverage"].values()):
        audit.coverage[target] = set()

    partial = limit is not None
    seen_keys: set[tuple[Any, ...]] = set()
    person_ids = ids_by_entity.get("person", set())
    character_ids = ids_by_entity.get("character", set())
    started = time.perf_counter()

    for _, record in iter_jsonl(path, file_stats, limit=limit):
        key = tuple(record.get(field_name) for field_name in spec["natural_key"])
        if key in seen_keys:
            audit.duplicate_count += 1
            if len(audit.duplicate_examples) < 5:
                audit.duplicate_examples.append(
                    {field_name: record.get(field_name) for field_name in spec["natural_key"]}
                )
        else:
            seen_keys.add(key)

        for name in spec["counters"]:
            value = record.get(name)
            if isinstance(value, (list, dict)):
                continue
            audit.counters[name][value] += 1
        for name in spec["empty_fields"]:
            if _is_missing(record.get(name)):
                audit.empty_counts[name] += 1

        if not partial:
            for field_name, target_entity in spec["endpoints"]:
                value = record.get(field_name)
                target_ids = ids_by_entity.get(target_entity, set())
                if not isinstance(value, int) or isinstance(value, bool) or value not in target_ids:
                    entry = audit.orphans[field_name]
                    entry["count"] += 1
                    if value not in entry["examples"] and len(entry["examples"]) < 5:
                        entry["examples"].append(value)

        for field_name, target in spec["coverage"].items():
            value = record.get(field_name)
            if isinstance(value, int) and not isinstance(value, bool):
                audit.coverage[target].add(value)

        if spec["name"] == "person-relations" and not partial:
            person_type = record.get("person_type")
            for field_name in ("person_id", "related_person_id"):
                value = record.get(field_name)
                if not isinstance(value, int) or isinstance(value, bool):
                    membership = "invalid"
                else:
                    in_person = value in person_ids
                    in_character = value in character_ids
                    if in_person and in_character:
                        membership = "both"
                    elif in_person:
                        membership = "person_only"
                    elif in_character:
                        membership = "character_only"
                    else:
                        membership = "neither"
                audit.typed_membership[(person_type, field_name, membership)] += 1

        if spec["name"] == "subject-relations":
            left = record.get("subject_id")
            right = record.get("related_subject_id")
            if isinstance(left, int) and left == right:
                audit.counters["self_loop"][left] += 1

    file_stats.elapsed_sec = time.perf_counter() - started
    if verbose:
        print(
            f"  {spec['name']:<19} {file_stats.valid_records:>9,} records  "
            f"{file_stats.size_bytes / 1024 / 1024:>8.1f} MiB  "
            f"{file_stats.elapsed_sec:>6.1f}s",
            file=sys.stderr,
            flush=True,
        )
    return audit


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------


def _esc(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _fmt_value(value: Any) -> str:
    if isinstance(value, bool):
        return "是" if value else "否"
    if value is None:
        return "—"
    return _esc(value)


def _code(value: Any) -> str:
    if value is None:
        return "—"
    return f"`{_fmt_value(value)}`"


def _fmt_int(value: int) -> str:
    return f"{value:,}"


def _pct(part: int, whole: int) -> str:
    if whole <= 0:
        return "—"
    return f"{part / whole * 100:.2f}%"


def _table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        lines.append("| " + " | ".join(_esc(cell) for cell in row) + " |")
    return lines


def _sorted_items(counter: Counter, top: int | None = None) -> list[tuple[Any, int]]:
    items = list(counter.items())
    items.sort(key=lambda pair: (-pair[1], str(pair[0])))
    return items[:top] if top else items


def _distribution_rows(
    counter: Counter,
    *,
    total: int,
    top: int | None = None,
    examples: dict[Any, list[str]] | None = None,
) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for value, count in _sorted_items(counter, top):
        row: list[Any] = [_code(value), _fmt_int(count), _pct(count, total)]
        if examples is not None:
            names = examples.get(value, [])
            row.append("、".join(names) if names else "—")
        rows.append(row)
    return rows


def _missing_table(audit: EntityAudit) -> list[str]:
    total = audit.file.valid_records
    rows = []
    for field_name, count in audit.missing.items():
        rows.append([_code(field_name), _fmt_int(count), _pct(count, total)])
    return _table(["字段", "缺失/空值条数", "占有效记录"], rows)


def _length_table(audit: EntityAudit) -> list[str]:
    rows = []
    for field_name, stats in audit.lengths.items():
        summary = stats.summary()
        rows.append(
            [
                _code(field_name),
                _fmt_int(summary["non_empty"]),
                _fmt_int(summary["empty"]),
                _fmt_int(summary["min"]),
                _fmt_int(summary["p50"]),
                _fmt_int(summary["p90"]),
                _fmt_int(summary["p99"]),
                _fmt_int(summary["max"]),
                summary["mean"],
            ]
        )
    return _table(
        ["字段", "非空", "空", "最短", "P50", "P90", "P99", "最长", "平均长度"],
        rows,
    )


def _file_stats_rows(entities: dict[str, EntityAudit], relations: dict[str, RelationAudit]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    ordered = list(entities.values()) + list(relations.values())
    for audit in ordered:
        stats = audit.file
        rows.append(
            [
                f"`{stats.name}`",
                f"{stats.size_bytes / 1024 / 1024:.1f}",
                _fmt_int(stats.total_lines),
                _fmt_int(stats.valid_records),
                _fmt_int(stats.blank_lines),
                _fmt_int(stats.parse_errors),
                _fmt_int(stats.decode_errors),
                _fmt_int(stats.schema_errors),
                f"{stats.elapsed_sec:.1f}",
            ]
        )
    return rows


def _language_table(audit: EntityAudit) -> list[str]:
    rows = []
    for key, count in _sorted_items(audit.language):
        rows.append([LANG_LABELS.get(key, key), _fmt_int(count), _pct(count, audit.language_total)])
    return _table(["文本类别", "条数", "占非空样本"], rows)


def _infobox_section(audit: EntityAudit) -> list[str]:
    infobox = audit.infobox
    assert infobox is not None
    lines: list[str] = []
    rows = [
        [
            f"`{audit.entity}`",
            _fmt_int(infobox.non_empty),
            _fmt_int(infobox.parsed),
            _fmt_int(infobox.failed),
            _pct(infobox.parsed, infobox.non_empty),
            _fmt_int(infobox.alias_docs),
            _fmt_int(infobox.multiline_alias_docs),
            (
                f"{infobox.alias_total / infobox.alias_docs:.2f}"
                if infobox.alias_docs
                else "—"
            ),
        ]
    ]
    lines.extend(
        _table(
            [
                "实体",
                "非空 infobox",
                "解析成功",
                "解析无结果",
                "解析成功率",
                "含别名文档",
                "多行别名格式",
                "平均别名数",
            ],
            rows,
        )
    )
    lines.append("")
    lines.append(
        "- “含别名文档”按 `parse_aliases` 返回非空计算；"
        "很多记录虽然存在 `别名` 字段，但内容为空块（`别名={\\n}`），不计入别名覆盖。"
    )
    lines.append("")

    key_rows = []
    for key, count in _sorted_items(infobox.keys, 25):
        key_rows.append([_code(key), _fmt_int(count), _pct(count, infobox.parsed)])
    lines.append("**高频 infobox 字段（按出现文档数）**")
    lines.append("")
    lines.extend(_table(["字段", "文档数", "占解析成功"], key_rows))
    lines.append("")

    name_rows = []
    for key in NAME_KEYS:
        if key in infobox.name_key_docs:
            count = infobox.name_key_docs[key]
            name_rows.append([_code(key), _fmt_int(count), _pct(count, infobox.parsed)])
    if name_rows:
        lines.append("**名称/别名相关字段出现情况**")
        lines.append("")
        lines.extend(_table(["字段", "文档数", "占解析成功"], name_rows))
        lines.append("")

    if audit.entity == "subject":
        extra = infobox.name_key_docs.get("__name_cn_missing_but_infobox_cn__", 0)
        lines.append(
            f"- `name_cn` 为空但 infobox 中存在中文名字段的作品：**{_fmt_int(extra)}** 条，"
            "P3 构建作品文档时应合并 `name_cn`、`name` 与 infobox 中文名/别名。"
        )
        lines.append("")

    if infobox.alias_examples:
        lines.append("**别名解析示例**")
        lines.append("")
        for record_id, aliases in infobox.alias_examples:
            lines.append(f"- `#{record_id}`：{', '.join(aliases)}")
        lines.append("")

    if infobox.raw_example:
        example = infobox.raw_example
        lines.append("**infobox 原文与解析结果示例**")
        lines.append("")
        lines.append(f"- 实体 ID：`{example['id']}`")
        lines.append(f"- 解析出的字段：{', '.join(example['keys'])}")
        lines.append(f"- 解析出的别名：{', '.join(example['aliases']) if example['aliases'] else '—'}")
        lines.append("")
        lines.append("```text")
        lines.append(example["raw"])
        lines.append("```")
        lines.append("")

    if infobox.failure_examples:
        lines.append("**无法解析出字段的 infobox 样例**")
        lines.append("")
        lines.append("```text")
        for failure in infobox.failure_examples:
            lines.append(f"#{failure['id']}: {failure['raw']}")
        lines.append("```")
        lines.append("")

    return lines


def _relationship_coverage_rows(
    entities: dict[str, EntityAudit], relations: dict[str, RelationAudit]
) -> list[list[Any]]:
    subject_ids = entities["subject"].ids
    character_ids = entities["character"].ids
    person_ids = entities["person"].ids

    subject_characters = relations["subject-characters"].coverage
    subject_persons = relations["subject-persons"].coverage
    subject_relations = relations["subject-relations"].coverage
    person_characters = relations["person-characters"].coverage

    subjects_with_characters = subject_characters.get("subjects", set())
    subjects_with_persons = subject_persons.get("subjects", set())
    subjects_with_relations = subject_relations.get("subjects", set())
    characters_with_subjects = subject_characters.get("characters", set())
    persons_with_subjects = subject_persons.get("persons", set())
    characters_with_persons = person_characters.get("characters", set())

    episode_audit = entities["episode"]
    subjects_with_episodes = episode_audit.extra.get("episode_subject_ids", set())
    linked_episode_ids = episode_audit.extra.get("linked_ids", set())
    episode_ids = episode_audit.ids

    rows: list[list[Any]] = [
        [
            "有作品记录但没有角色关联的作品",
            _fmt_int(len(subject_ids - subjects_with_characters)),
            _pct(len(subject_ids - subjects_with_characters), len(subject_ids)),
            _sample_ids(subject_ids - subjects_with_characters),
        ],
        [
            "有作品记录但没有章节的作品",
            _fmt_int(len(subject_ids - subjects_with_episodes)),
            _pct(len(subject_ids - subjects_with_episodes), len(subject_ids)),
            _sample_ids(subject_ids - subjects_with_episodes),
        ],
        [
            "有作品记录但没有制作人员关联的作品",
            _fmt_int(len(subject_ids - subjects_with_persons)),
            _pct(len(subject_ids - subjects_with_persons), len(subject_ids)),
            _sample_ids(subject_ids - subjects_with_persons),
        ],
        [
            "有作品记录但没有作品间关系的作品",
            _fmt_int(len(subject_ids - subjects_with_relations)),
            _pct(len(subject_ids - subjects_with_relations), len(subject_ids)),
            _sample_ids(subject_ids - subjects_with_relations),
        ],
        [
            "没有关联任何作品的角色",
            _fmt_int(len(character_ids - characters_with_subjects)),
            _pct(len(character_ids - characters_with_subjects), len(character_ids)),
            _sample_ids(character_ids - characters_with_subjects),
        ],
        [
            "没有关联任何作品的人物",
            _fmt_int(len(person_ids - persons_with_subjects)),
            _pct(len(person_ids - persons_with_subjects), len(person_ids)),
            _sample_ids(person_ids - persons_with_subjects),
        ],
        [
            "没有人物关联的角色",
            _fmt_int(len(character_ids - characters_with_persons)),
            _pct(len(character_ids - characters_with_persons), len(character_ids)),
            _sample_ids(character_ids - characters_with_persons),
        ],
        [
            "没有所属作品的章节",
            _fmt_int(len(episode_ids - linked_episode_ids)),
            _pct(len(episode_ids - linked_episode_ids), len(episode_ids)),
            _sample_ids(episode_ids - linked_episode_ids),
        ],
    ]
    return rows


def _sample_ids(ids: set[int], n: int = 5) -> str:
    if not ids:
        return "—"
    return ", ".join(f"`{value}`" for value in sorted(ids)[:n])


# ---------------------------------------------------------------------------
# Report composition
# ---------------------------------------------------------------------------


def build_report(
    entities: dict[str, EntityAudit],
    relations: dict[str, RelationAudit],
    *,
    meta: dict[str, Any],
    raw_dir: Path,
) -> str:
    subject = entities["subject"]
    character = entities["character"]
    person = entities["person"]
    episode = entities["episode"]

    total_records = sum(audit.file.valid_records for audit in entities.values())
    total_relations = sum(audit.file.valid_records for audit in relations.values())
    total_parse_errors = sum(
        audit.file.bad_lines for audit in list(entities.values()) + list(relations.values())
    )

    subject_characters = relations["subject-characters"].coverage
    subjects_with_characters = subject_characters.get("subjects", set())
    characters_with_subjects = subject_characters.get("characters", set())

    summary_ids = subject.extra.get("summary_ids", set())
    summary_safe_ids = subject.extra.get("summary_safe_ids", set())
    character_summary_ids = character.extra.get("summary_ids", set())
    episode_description_ids = episode.extra.get("description_ids", set())
    episode_linked_ids = episode.extra.get("linked_ids", set())

    character_summary_linked = len(character_summary_ids & characters_with_subjects)
    episode_desc_linked = len(episode_description_ids & episode_linked_ids)
    narrative_ratio = (
        episode.quality["narrative"] / episode.quality_sample_size
        if episode.quality_sample_size
        else 0.0
    )
    mixed_ratio = (
        episode.quality["mixed"] / episode.quality_sample_size
        if episode.quality_sample_size
        else 0.0
    )

    lines: list[str] = []
    add = lines.append

    add("# Bangumi Archive 数据审计报告")
    add("")
    add("> 本报告由 `scripts/audit_data.py` 自动生成，请勿手工编辑。")
    add("")
    add(f"- 生成时间：{meta['generated_at']}")
    add(f"- 执行命令：`{meta['command']}`")
    add(f"- 数据目录：`{raw_dir}`")
    if meta.get("limit") is not None:
        add(
            f"- **冒烟模式：每个文件仅读取前 {_fmt_int(meta['limit'])} 行，"
            "关联完整性与覆盖统计不可用于全量结论。**"
        )
    else:
        add("- 审计范围：全部 9 份文件、全量行数（未使用采样替代行数统计）。")
    add(f"- 运行耗时：{meta['elapsed_sec']:.1f} 秒")
    if meta.get("peak_rss_mb") is not None:
        add(f"- 进程峰值内存：{meta['peak_rss_mb']:.1f} MiB")
    add(f"- Python：{meta['python']}；平台：{meta['platform']}")
    add("- 原始数据只读：审计以二进制只读方式打开文件，未写入、未修改 `data/raw/`。")
    add("")

    add("## 0. 结论摘要")
    add("")
    add(
        f"- 4 个主实体文件共 **{_fmt_int(total_records)}** 条有效记录；"
        f"5 个关联文件共 **{_fmt_int(total_relations)}** 条有效记录。"
    )
    add(f"- 9 份文件合计解析/解码/结构异常 **{_fmt_int(total_parse_errors)}** 行。")
    add(
        f"- 作品 {_fmt_int(len(subject.ids))} 条，其中简介非空 "
        f"{_fmt_int(subject.lengths['summary'].non_empty)} 条；"
        f"非 NSFW 且简介非空 {_fmt_int(len(summary_safe_ids))} 条。"
    )
    add(
        f"- 角色 {_fmt_int(len(character.ids))} 条，其中简介非空 "
        f"{_fmt_int(character.lengths['summary'].non_empty)} 条，"
        f"可映射到作品且有简介 {_fmt_int(character_summary_linked)} 条。"
    )
    add(
        f"- 章节 {_fmt_int(len(episode.ids))} 条，描述非空 "
        f"{_fmt_int(episode.lengths['description'].non_empty)} 条；"
        f"抽样中疑似剧情占 {narrative_ratio * 100:.1f}%、"
        f"制作信息+剧情混合占 {mixed_ratio * 100:.1f}%。"
    )
    if subject.infobox is not None:
        extra_cn = subject.infobox.name_key_docs.get("__name_cn_missing_but_infobox_cn__", 0)
        add(
            f"- 作品 `name_cn` 为空但 infobox 含中文名的有 "
            f"{_fmt_int(extra_cn)} 条，别名可直接从 infobox 的 `别名` 字段解析。"
        )
    add("- 数值码值（`type`、`platform`、`relation_type`、`position`、`role` 等）"
        "仅报告分布与样例，未擅自解释业务含义。")
    add("")

    add("## 1. 文件级统计")
    add("")
    lines.extend(
        _table(
            [
                "文件",
                "大小 (MiB)",
                "总行数",
                "有效记录",
                "空行",
                "JSON 失败",
                "解码失败",
                "非对象",
                "耗时 (s)",
            ],
            _file_stats_rows(entities, relations),
        )
    )
    add("")
    add(
        "> 有效记录 + 空行 + 各类异常 = 总行数。文件哈希见附录 A；"
        "使用 `--limit` 时哈希不计算，以保证部分读取不会被误当作全量校验值。"
    )
    add("")

    add("## 2. 主实体完整性")
    add("")
    for audit in (subject, character, person, episode):
        stats = audit.file
        add(f"### 2.{list(entities).index(audit.entity) + 1} `{stats.name}`")
        add("")
        add(
            f"- 有效记录：{_fmt_int(stats.valid_records)}；唯一 ID："
            f"{_fmt_int(len(audit.ids))}；重复 ID：{_fmt_int(audit.duplicate_id_count)}；"
            f"非法/缺失 ID：{_fmt_int(audit.invalid_id_count)}"
        )
        if audit.duplicate_id_examples:
            add(f"- 重复 ID 示例：{', '.join(f'`{value}`' for value in audit.duplicate_id_examples)}")
        add("")
        add("**字段缺失/空值**")
        add("")
        lines.extend(_missing_table(audit))
        add("")
        add("**文本长度（字符数）**")
        add("")
        lines.extend(_length_table(audit))
        add("")

    add("## 3. 字段分布")
    add("")
    add("### 3.1 作品")
    add("")
    add("**type 分布（码值含义未解释）**")
    add("")
    lines.extend(
        _table(
            ["type", "数量", "占比", "示例作品"],
            _distribution_rows(subject.counters["type"], total=len(subject.ids), examples=subject.code_examples["type"]),
        )
    )
    add("")
    add("**platform 分布（前 20）**")
    add("")
    lines.extend(
        _table(
            ["platform", "数量", "占比", "示例作品"],
            _distribution_rows(
                subject.counters["platform"],
                total=len(subject.ids),
                top=20,
                examples=subject.code_examples["platform"],
            ),
        )
    )
    add("")
    add("**NSFW 标记**")
    add("")
    lines.extend(
        _table(
            ["nsfw", "数量", "占比"],
            _distribution_rows(subject.counters["nsfw"], total=len(subject.ids)),
        )
    )
    add("")
    add("**年份分布（`date` 前四位，前 20）**")
    add("")
    lines.extend(
        _table(
            ["年份", "数量", "占比"],
            _distribution_rows(subject.counters["date_year"], total=len(subject.ids), top=20),
        )
    )
    add("")
    add("**标签频次（前 30）**")
    add("")
    tag_rows = [
        [_esc(name), _fmt_int(count)]
        for name, count in _sorted_items(subject.counters["tags"], 30)
    ]
    lines.extend(_table(["标签", "作品数"], tag_rows))
    add("")

    add("### 3.2 角色")
    add("")
    add("**role 分布（码值含义未解释）**")
    add("")
    lines.extend(
        _table(
            ["role", "数量", "占比", "示例角色"],
            _distribution_rows(character.counters["role"], total=len(character.ids), examples=character.code_examples["role"]),
        )
    )
    add("")

    add("### 3.3 人物")
    add("")
    add("**type 分布（码值含义未解释）**")
    add("")
    lines.extend(
        _table(
            ["type", "数量", "占比", "示例人物"],
            _distribution_rows(person.counters["type"], total=len(person.ids), examples=person.code_examples["type"]),
        )
    )
    add("")
    add("**career 分布（前 20）**")
    add("")
    career_rows = [
        [_esc(name), _fmt_int(count)]
        for name, count in _sorted_items(person.counters["career"], 20)
    ]
    lines.extend(_table(["career", "人物数"], career_rows))
    add("")
    add("**career 列表长度分布（10 表示 >=10）**")
    add("")
    lines.extend(
        _table(
            ["列表长度", "人物数", "占比"],
            _distribution_rows(person.counters["career_len"], total=len(person.ids)),
        )
    )
    add("")

    add("### 3.4 章节")
    add("")
    add("**type 分布（码值含义未解释）**")
    add("")
    lines.extend(
        _table(
            ["type", "数量", "占比", "示例章节"],
            _distribution_rows(episode.counters["type"], total=len(episode.ids), examples=episode.code_examples["type"]),
        )
    )
    add("")
    add("**duration 分布（前 20）**")
    add("")
    duration_rows = [
        [_code(name), _fmt_int(count)]
        for name, count in _sorted_items(episode.counters["duration"], 20)
    ]
    lines.extend(_table(["duration", "章节数"], duration_rows))
    add("")
    add("**airdate 年份分布（前 20）**")
    add("")
    lines.extend(
        _table(
            ["年份", "数量", "占比"],
            _distribution_rows(episode.counters["airdate_year"], total=len(episode.ids), top=20),
        )
    )
    add("")

    add("## 4. 关联表完整性与分布")
    add("")
    for relation in relations.values():
        stats = relation.file
        add(f"### 4.{list(relations).index(relation.name) + 1} `{stats.name}`")
        add("")
        add(
            f"- 有效记录：{_fmt_int(stats.valid_records)}；"
            f"按自然键去重后：{_fmt_int(stats.valid_records - relation.duplicate_count)}；"
            f"重复自然键：{_fmt_int(relation.duplicate_count)}"
        )
        add(f"- 自然键：`{', '.join(relation.natural_key)}`")
        if relation.duplicate_examples:
            add(f"- 重复示例：`{json.dumps(relation.duplicate_examples[:3], ensure_ascii=False)}`")
        add("")
        if meta.get("limit") is None and relation.name == "person-relations":
            add("**person_type 命名空间下的 ID 归属**")
            add("")
            typed_rows = []
            person_types = sorted(
                {str(key[0]) for key in relation.typed_membership},
                key=lambda value: str(value),
            )
            for person_type in person_types:
                for field_name in ("person_id", "related_person_id"):
                    typed_rows.append(
                        [
                            _code(person_type),
                            _code(field_name),
                            _fmt_int(relation.typed_membership.get((person_type, field_name, "person_only"), 0)),
                            _fmt_int(relation.typed_membership.get((person_type, field_name, "character_only"), 0)),
                            _fmt_int(relation.typed_membership.get((person_type, field_name, "both"), 0)),
                            _fmt_int(relation.typed_membership.get((person_type, field_name, "neither"), 0)),
                            _fmt_int(relation.typed_membership.get((person_type, field_name, "invalid"), 0)),
                        ]
                    )
            lines.extend(
                _table(
                    [
                        "person_type",
                        "字段",
                        "仅 person 存在",
                        "仅 character 存在",
                        "两边都存在",
                        "两边都不存在",
                        "无效 ID",
                    ],
                    typed_rows,
                )
            )
            add("")
            overlap = len(entities["person"].ids & entities["character"].ids)
            add(
                f"> person 与 character 的 ID 集合有 **{_fmt_int(overlap)}** 个重叠，"
                "因此仅凭 ID 无法在两边都存在的记录上唯一判定命名空间。"
                "本表只报告归属分布，不把任一解释下的未命中直接称为孤儿记录。"
            )
        elif meta.get("limit") is None:
            orphan_rows = []
            for field_name, entry in relation.orphans.items():
                orphan_rows.append(
                    [
                        _code(field_name),
                        _code(entry["entity"]),
                        _fmt_int(entry["count"]),
                        ", ".join(_code(value) for value in entry["examples"]) or "—",
                    ]
                )
            lines.extend(
                _table(
                    ["外键字段", "目标实体", "孤儿记录数", "示例值"],
                    orphan_rows,
                )
            )
        else:
            add("> `--limit` 模式下跳过孤儿记录判定。")
        add("")
        for name, counter in relation.counters.items():
            if not counter:
                continue
            add(f"**{name} 分布**")
            add("")
            if name == "position":
                rows = _distribution_rows(counter, total=stats.valid_records, top=20)
            elif name == "relation_type":
                rows = _distribution_rows(counter, total=stats.valid_records, top=20)
            else:
                rows = _distribution_rows(counter, total=stats.valid_records)
            lines.extend(_table([name, "数量", "占比"], rows))
            add("")
        for field_name, count in relation.empty_counts.items():
            add(
                f"- `{field_name}` 缺失/空值：{_fmt_int(count)} "
                f"（{_pct(count, stats.valid_records)}）"
            )
        if relation.name == "subject-relations":
            self_loops = sum(relation.counters.get("self_loop", Counter()).values())
            add(f"- 自关联（`subject_id == related_subject_id`）：{_fmt_int(self_loops)} 条")
        add("")

    add("## 5. 关联覆盖")
    add("")
    if meta.get("limit") is None:
        lines.extend(
            _table(
                ["覆盖指标", "数量", "占比", "示例 ID"],
                _relationship_coverage_rows(entities, relations),
            )
        )
        add("")
        add(
            "> “没有章节的作品”依据 `episode.subject_id` 与作品 ID 集合计算；"
            "其余指标依据关联表。孤儿关联已在上节单独统计。"
        )
    else:
        add("> `--limit` 模式下关联覆盖无意义，已跳过。")
    add("")

    add("## 6. infobox 格式与解析")
    add("")
    for audit in (subject, character, person):
        add(f"### 6.{list(entities).index(audit.entity) + 1} `{audit.file.name}`")
        add("")
        lines.extend(_infobox_section(audit))

    add("## 7. 文本语言覆盖")
    add("")
    add(
        "> 语言分类基于 Unicode 区间与一组常用简繁专用字对做近似判断；"
        "结果用于覆盖率评估，不适合作为逐条语言判定依据。"
    )
    add("")
    for audit in (subject, character, person):
        add(f"### 7.{list(entities).index(audit.entity) + 1} `{audit.file.name}` 的 `summary`（全量）")
        add("")
        lines.extend(_language_table(audit))
        add("")
    add(f"### 7.4 `{episode.file.name}` 的 `description`（抽样 {_fmt_int(episode.quality_sample_size)} 条）")
    add("")
    lines.extend(_language_table(episode))
    add("")

    add("## 8. 章节描述质量")
    add("")
    add(
        f"对 {_fmt_int(episode.quality_sample_size)} 条章节记录做固定种子蓄水池抽样，"
        "按行判断“制作人员/制作信息”与“剧情文本”。判定规则："
        "行内出现 `シナリオ/脚本/コンテ/演出/作画監督/キャスト/スタッフ/主題歌` 等标记、"
        "长度不超过 200 且不含句末标点时视为制作信息；"
        "含句末标点或长度 >= 12 的其他行视为叙事文本。"
    )
    add("")
    quality_rows = []
    for key, count in _sorted_items(episode.quality):
        quality_rows.append(
            [QUALITY_LABELS.get(key, key), _fmt_int(count), _pct(count, episode.quality_sample_size)]
        )
    lines.extend(_table(["类别", "抽样条数", "占抽样"], quality_rows))
    add("")
    for category, examples in episode.quality_examples.items():
        if not examples:
            continue
        add(f"**{QUALITY_LABELS.get(category, category)} 示例**")
        add("")
        for example in examples:
            add(f"- {example}")
        add("")
    add(
        "> 该分类只用于确定 P3 的过滤规则，不能直接当作人工标注。"
        "后续应通过评测验证过滤是否过严。"
    )
    add("")

    add("## 9. 首期索引范围建议")
    add("")
    add(
        "以下范围基于本次实测数量给出。P3/P4 仍应按规划先做小规模闭环，"
        "再用评测决定是否扩大。"
    )
    if meta.get("limit") is not None:
        add("")
        add(
            "> **冒烟模式：本节数量仅来自每个文件的前 N 行，不能作为全量索引范围依据。**"
        )
    add("")
    add("### 9.1 作品文档")
    add("")
    add(
        f"- 实测作品 {_fmt_int(len(subject.ids))} 条，简介非空 "
        f"{_fmt_int(subject.lengths['summary'].non_empty)} 条。"
    )
    add(
        f"- 建议首期纳入“简介非空”的作品；若需要进一步控量，先取非 NSFW 且简介非空的 "
        f"{_fmt_int(len(summary_safe_ids))} 条做 1 万条以内的索引实验，再逐步扩容。"
    )
    add(
        "- 文档必须合并 `name`、`name_cn`、infobox 中文名和 `别名`，"
        "并把 `nsfw`、`type`、`platform`、年份放入元数据，不要用其替代文本。"
    )
    add("")
    add("### 9.2 角色文档")
    add("")
    add(
        f"- 实测角色 {_fmt_int(len(character.ids))} 条，简介非空 "
        f"{_fmt_int(character.lengths['summary'].non_empty)} 条，"
        f"可映射到作品且简介非空 {_fmt_int(character_summary_linked)} 条。"
    )
    add(
        "- 建议首期纳入“简介非空且能通过 `subject_characters` 映射到作品”的角色；"
        "没有简介但名称/别名可靠的角色可后续补充。"
    )
    add("")
    add("### 9.3 章节文档")
    add("")
    add(
        f"- 实测章节 {_fmt_int(len(episode.ids))} 条，描述非空 "
        f"{_fmt_int(episode.lengths['description'].non_empty)} 条，"
        f"所属作品存在 {_fmt_int(len(episode_linked_ids))} 条；"
        f"描述非空且所属作品存在 {_fmt_int(episode_desc_linked)} 条。"
    )
    add(
        f"- 抽样中疑似剧情 {narrative_ratio * 100:.1f}%、制作信息+剧情混合 "
        f"{mixed_ratio * 100:.1f}%。建议 P3 仅纳入通过质量规则且能映射到作品的章节，"
        "并保存过滤原因。"
    )
    add("")
    add("### 9.4 别名与实体关系")
    add("")
    add(
        f"- 作品 infobox 中可解析出别名的文档："
        f"{_fmt_int(subject.infobox.alias_docs if subject.infobox else 0)} 条。"
    )
    add(
        "- 角色和章节命中后必须通过 `subject_characters` 映射回作品，"
        "章节还要校验 `episode.subject_id`，避免把孤儿记录当作证据。"
    )
    add("")

    add("## 10. 对 P2 结构化库的 Schema 建议")
    add("")
    add("本建议来自实际字段，不删除原始业务字段：")
    add("")
    add("```text")
    add("subject(id PK, type, name, name_cn, infobox_raw, platform, summary, nsfw, tags_json,")
    add("        meta_tags_json, score, score_details_json, rank, date, favorite_json, series,")
    add("        source_file)")
    add("episode(id PK, name, name_cn, description, airdate, disc, duration, subject_id FK,")
    add("        sort, type, source_file)")
    add("character(id PK, role, name, infobox_raw, summary, comments, collects, source_file)")
    add("person(id PK, name, type, career_json, infobox_raw, summary, comments, collects, source_file)")
    add("")
    add("subject_characters(subject_id, character_id, type, ord, PRIMARY KEY(subject_id, character_id))")
    add("subject_persons(subject_id, person_id, position, appear_eps,")
    add("                PRIMARY KEY(subject_id, person_id, position))")
    add("subject_relations(subject_id, relation_type, related_subject_id, ord,")
    add("                   PRIMARY KEY(subject_id, related_subject_id, relation_type))")
    add("person_characters(person_id, subject_id, character_id, type, summary,")
    add("                    PRIMARY KEY(person_id, subject_id, character_id))")
    add("person_relations(person_type, person_id, related_person_id, relation_type,")
    add("                 spoiler, ended,")
    add("                 PRIMARY KEY(person_type, person_id, related_person_id, relation_type))")
    add("```")
    add("")
    add("- 索引：`episode(subject_id)`、`subject_characters(character_id)`、")
    add("  `subject_persons(person_id)`、`subject_relations(related_subject_id)`、")
    add("  `person_characters(character_id)`、`person_relations(person_type, person_id)`。")
    add("- 原始 `infobox` 以文本形式保留，解析结果另行生成，不覆盖原文。")
    add("- 代码含义未确认的字段（`type`/`platform`/`position`/`relation_type`/`person_type`/`role`）")
    add("  先按整数或原值存储，另建码值说明表，不写入推测性标签。")
    add(
        "- `person_relations` 不应直接对 person/character 建强外键：两种 ID 空间存在大量重叠，"
        "P2 应以 `person_type` 作为命名空间保存原值，并在文档层说明其含义仍需确认。"
    )
    add("- 建议为每个表保留 `source_file`，配合实体 ID 即可追溯回原始 JSONL。")
    add("")

    add("## 11. 数据异常与待确认问题")
    add("")
    if meta.get("limit") is not None:
        add(
            f"- 冒烟模式仅读取每个文件的前 {_fmt_int(meta['limit'])} 行，"
            "本节及全文的异常数量、覆盖率和范围建议不代表全量数据。"
        )
    elif total_parse_errors:
        add(f"- 9 份文件共有 {_fmt_int(total_parse_errors)} 行无法解析或不是对象，详见各文件异常示例。")
    else:
        add("- 本次全量读取未发现 JSON 解析、解码或非对象记录错误。")
    add(
        "- `episode.description` 混有制作名单与剧情文本；P3 必须按质量规则过滤并保留过滤原因。"
    )
    add(
        "- `person_relations.person_type` 取值为 `crt` / `prsn`；抽样中 `crt` 记录的名称通常对应角色，"
        "但 person 与 character 的 ID 集合有 "
        f"{_fmt_int(len(entities['person'].ids & entities['character'].ids))} 个重叠，"
        "仅凭 ID 不能唯一判定。P2 应按 `person_type` 分命名空间保存，不做跨表强外键。"
    )
    add(
        "- 关联表自然键重复可能来自原始数据重复导出，重复数量已在第 4 节列出；"
        "P2 导入必须使用唯一约束或显式去重策略。"
    )
    add(
        "- 繁体/简体分类基于启发式字符对，仅用于覆盖率；如需要精确统计，P3 可引入 OpenCC 做对照实验。"
    )
    add("")

    add("## 附录 A：文件 SHA256 与异常示例")
    add("")
    add("### A.1 SHA256")
    add("")
    add("```text")
    for audit in list(entities.values()) + list(relations.values()):
        stats = audit.file
        digest = stats.sha256 if stats.sha256 else "（--limit 模式未计算）"
        add(f"{digest}  {stats.name}")
    add("```")
    add("")
    add("### A.2 异常行示例")
    add("")
    error_entries = [
        (audit.file.name, example)
        for audit in list(entities.values()) + list(relations.values())
        for example in audit.file.error_examples
    ]
    if error_entries:
        add("```text")
        for filename, example in error_entries[:30]:
            add(
                f"{filename}:{example['line']} [{example['kind']}] "
                f"{example['error']} | {example['sample']}"
            )
        add("```")
    else:
        add("无。")
    add("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_audit(
    *,
    raw_dir: Path = RAW_DIR,
    output_path: Path = DOCS_DIR / "data-audit.md",
    limit: int | None = None,
    sample_size: int = 3000,
    seed: int = 42,
    command: str = "python scripts/audit_data.py",
    verbose: bool = True,
) -> dict[str, Any]:
    started_at = datetime.now().astimezone()
    started = time.perf_counter()
    raw_dir = Path(raw_dir)
    output_path = Path(output_path)

    missing_files = [filename for filename in ALL_DATA_FILES if not (raw_dir / filename).exists()]
    if missing_files:
        raise FileNotFoundError(
            "缺少原始数据文件：" + ", ".join(str(raw_dir / name) for name in missing_files)
        )

    if verbose:
        print(f"审计目录：{raw_dir}", file=sys.stderr, flush=True)
        print(f"模式：{'冒烟（--limit）' if limit is not None else '全量'}", file=sys.stderr, flush=True)

    entities: dict[str, EntityAudit] = {}
    for entity, filename in MAIN_ENTITY_FILES.items():
        if verbose:
            print(f"[主实体] {entity}", file=sys.stderr, flush=True)
        entities[entity] = audit_entity(
            entity,
            raw_dir / filename,
            limit=limit,
            sample_size=sample_size,
            seed=seed,
            foreign_ids={"subject": entities["subject"].ids} if entity == "episode" else None,
            verbose=verbose,
        )

    ids_by_entity = {entity: audit.ids for entity, audit in entities.items()}

    relations: dict[str, RelationAudit] = {}
    for spec in RELATION_SPECS:
        if verbose:
            print(f"[关联表] {spec['name']}", file=sys.stderr, flush=True)
        relations[spec["name"]] = audit_relation(
            spec,
            raw_dir / spec["filename"],
            ids_by_entity=ids_by_entity,
            limit=limit,
            verbose=verbose,
        )

    elapsed = time.perf_counter() - started
    meta = {
        "generated_at": started_at.isoformat(timespec="seconds"),
        "command": command,
        "limit": limit,
        "elapsed_sec": elapsed,
        "peak_rss_mb": peak_rss_mb(),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    report = build_report(entities, relations, meta=meta, raw_dir=raw_dir)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    temporary_path.write_text(report, encoding="utf-8", newline="\n")
    os.replace(temporary_path, output_path)

    if verbose:
        print(f"报告已写入：{output_path}", file=sys.stderr, flush=True)
        print(f"总耗时：{elapsed:.1f} 秒", file=sys.stderr, flush=True)

    return {
        "output": str(output_path),
        "elapsed_sec": elapsed,
        "entities": {entity: audit.file.valid_records for entity, audit in entities.items()},
        "relations": {name: audit.file.valid_records for name, audit in relations.items()},
        "bad_lines": sum(
            audit.file.bad_lines for audit in list(entities.values()) + list(relations.values())
        ),
    }
