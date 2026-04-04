import json
import random
import re
import os
from datasets import load_dataset, concatenate_datasets

# Project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# ============================================================
# 长尾实体库 (200+ unique pairs for Low-Saliency experiments)
# 来源: 生物学、化学、地理、历史等专业领域的冷门实体
# ============================================================
LOW_SALIENCY_ENTITIES = [
    # 生物学 - 蛋白质与基因
    ("Zfyve26 protein", "Pseudo-Zfyve26"), ("BRCA2 exon 11", "BRCA2-X11"),
    ("Titin connectin", "Titin-Alt"), ("Opsonic receptor CR3", "CR3-Variant"),
    ("Claudin-5 junction", "Claudin-5X"), ("Aquaporin-4 channel", "AQP4-Beta"),
    ("Synaptotagmin-1", "Synaptotagmin-Z"), ("Ubiquitin ligase E3", "E3-Pseudo"),
    ("Caspase-9 initiator", "Caspase-9X"), ("Dystrophin protein", "Dystrophin-Alt"),
    ("Presenilin-1 complex", "Presenilin-1X"), ("Telomerase hTERT", "hTERT-Var"),
    ("Hemoglobin S variant", "HbS-Modified"), ("Ferritin light chain", "Ferritin-LC2"),
    ("Myosin heavy chain", "Myosin-HCX"), ("Actin filament G-actin", "G-actin-V2"),
    ("Keratin-14 filament", "Keratin-14X"), ("Collagen type IV", "Collagen-IV-Alt"),
    ("Elastin tropoelastin", "Elastin-TE2"), ("Fibronectin domain III", "FN-DIII-V"),
    ("Laminin alpha-5", "Laminin-A5X"), ("Integrin beta-1", "Integrin-B1-Var"),
    ("Cadherin-11 adhesion", "Cadherin-11X"), ("Selectin P-selectin", "P-selectin-V2"),
    ("Mucin MUC5AC", "MUC5AC-Alt"), ("Defensin alpha-1", "Defensin-A1X"),
    # 生物学 - 有机体与物种
    ("Apicoplast organelle", "Synthetic-Apicoplast"), ("Merytre-Hatshepsut", "Hatshepsut-D"),
    ("Boulengerula taitana", "Boulengerula-X"), ("Mnemiopsis leidyi", "Mnemiopsis-V2"),
    ("Tardigrade Ramazzottius", "Ramazzottius-X"), ("Deinococcus radiodurans", "Deinococcus-V2"),
    ("Pyrococcus furiosus", "Pyrococcus-Alt"), ("Thermococcus kodakarensis", "T-kodakarensis-X"),
    ("Halobacterium salinarum", "H-salinarum-V2"), ("Sulfolobus acidocaldarius", "S-acidocaldarius-X"),
    ("Naegleria fowleri", "Naegleria-V2"), ("Plasmodium vivax", "P-vivax-Alt"),
    ("Trypanosoma cruzi", "T-cruzi-X"), ("Leishmania donovani", "L-donovani-V2"),
    ("Giardia lamblia", "G-lamblia-Alt"), ("Entamoeba histolytica", "E-histolytica-X"),
    ("Trichomonas vaginalis", "T-vaginalis-V2"), ("Babesia microti", "B-microti-Alt"),
    ("Cryptosporidium parvum", "C-parvum-X"), ("Toxoplasma gondii", "T-gondii-V2"),
    ("Schistosoma mansoni", "S-mansoni-Alt"), ("Onchocerca volvulus", "O-volvulus-X"),
    ("Wuchereria bancrofti", "W-bancrofti-V2"), ("Brugia malayi", "B-malayi-Alt"),
    ("Ancylostoma duodenale", "A-duodenale-X"),
    # 化学 - 化合物与反应
    ("Buckminsterfullerene C60", "C60-Isomer"), ("Ferrocene metallocene", "Ferrocene-Alt"),
    ("Zeolite ZSM-5 catalyst", "ZSM-5-V2"), ("Grubbs catalyst Ru", "Grubbs-Ru-X"),
    ("Cisplatin Pt complex", "Cisplatin-V2"), ("Taxol paclitaxel side chain", "Taxol-SC-Alt"),
    ("Thalidomide R-enantiomer", "Thalidomide-RX"), ("Aspirin acetyl group", "Aspirin-AG-V2"),
    ("Penicillin beta-lactam", "Penicillin-BL-Alt"), ("Strychnine alkaloid", "Strychnine-X"),
    ("Capsaicin vanilloid", "Capsaicin-V2"), ("Caffeine methylxanthine", "Caffeine-MX-Alt"),
    ("Nicotine pyridine ring", "Nicotine-PR-X"), ("Morphine phenanthrene", "Morphine-PH-V2"),
    ("Cholesterol steroid", "Cholesterol-ST-Alt"), ("Dopamine catechol", "Dopamine-CC-X"),
    ("Serotonin indole ring", "Serotonin-IR-V2"), ("Melatonin acetamide", "Melatonin-AC-Alt"),
    ("Oxytocin nonapeptide", "Oxytocin-NP-X"), ("Vasopressin cyclic", "Vasopressin-CY-V2"),
    ("Cortisol glucocorticoid", "Cortisol-GC-Alt"), ("Aldosterone mineral", "Aldosterone-MN-X"),
    ("Testosterone androgen", "Testosterone-AG-V2"), ("Estradiol phenol ring", "Estradiol-PR-Alt"),
    ("Progesterone ketone", "Progesterone-KT-X"),
    # 地理 - 冷门地名
    ("Oymyakon settlement", "Oymyakon-Alt"), ("Grytviken station", "Grytviken-Base"),
    ("Ittoqqortoormiit village", "Ittoqqortoormiit-V2"), ("Tristan da Cunha island", "TDC-Variant"),
    ("Bouvet Island territory", "Bouvet-Alt"), ("Heard Island volcano", "Heard-V2"),
    ("Kerguelen Plateau", "Kerguelen-Alt"), ("Deception Island caldera", "Deception-V2"),
    ("Jan Mayen volcanic", "JanMayen-X"), ("Svalbard Longyearbyen", "Svalbard-Alt"),
    ("Franz Josef Land archipelago", "FJL-V2"), ("Novaya Zemlya island", "NovayaZemlya-Alt"),
    ("Wrangel Island reserve", "Wrangel-X"), ("Socotra Island biodiversity", "Socotra-V2"),
    ("Aldabra Atoll giant", "Aldabra-Alt"), ("Clipperton Island ring", "Clipperton-V2"),
    ("Palmyra Atoll reef", "Palmyra-X"), ("Midway Atoll albatross", "Midway-V2"),
    ("Wake Island territory", "Wake-Alt"), ("Johnston Atoll military", "Johnston-V2"),
    ("Navassa Island disputed", "Navassa-Alt"), ("Baker Island uninhabited", "Baker-V2"),
    ("Howland Island equatorial", "Howland-Alt"), ("Jarvis Island coral", "Jarvis-V2"),
    ("Kingman Reef submerged", "Kingman-Alt"),
    # 历史 - 冷门事件与人物
    ("Peloponnesian truce", "Peloponnesian-T2"), ("Defenestration of Prague", "Defenestration-V2"),
    ("Treaty of Tordesillas", "Tordesillas-Alt"), ("Edict of Fontainebleau", "Fontainebleau-X"),
    ("Peace of Augsburg", "Augsburg-V2"), ("Diet of Worms decree", "Worms-Alt"),
    ("Concordat of Bologna", "Bologna-X"), ("Pragmatic Sanction decree", "Sanction-V2"),
    ("Golden Bull charter", "GoldenBull-Alt"), ("Statute of Laborers", "Laborers-X"),
    ("Assize of Clarendon", "Clarendon-V2"), ("Constitutions of Melfi", "Melfi-Alt"),
    ("Capitulary of Quierzy", "Quierzy-X"), ("Donation of Pepin", "Pepin-V2"),
    ("Edict of Milan authority", "Milan-Alt"), ("Theodosian Code law", "Theodosian-X"),
    ("Justinian Digest legal", "Digest-V2"), ("Lex Hortensia plebiscite", "Hortensia-Alt"),
    ("Lex Aquilia damages", "Aquilia-X"), ("Twelve Tables codex", "TwelveTables-V2"),
    ("Senatus consultum decree", "SenatusC-Alt"), ("Lex Julia adultery", "Julia-X"),
    ("Lex Oppia sumptuary", "Oppia-V2"), ("Lex Voconia inheritance", "Voconia-Alt"),
    ("Lex Falcidia testament", "Falcidia-X"),
    # 矿物学与地质学
    ("Coesite silica polymorph", "Coesite-V2"), ("Stishovite high pressure", "Stishovite-Alt"),
    ("Ringwoodite olivine", "Ringwoodite-X"), ("Bridgmanite perovskite", "Bridgmanite-V2"),
    ("Davemaoite CaSiO3", "Davemaoite-Alt"), ("Wadsleyite spinel", "Wadsleyite-X"),
    ("Majorite garnet", "Majorite-V2"), ("Akimotoite ilmenite", "Akimotoite-Alt"),
    ("Ferropericlase mantle", "Ferropericlase-X"), ("Lingunite feldspar", "Lingunite-V2"),
    ("Hollandite KAlSi3O8", "Hollandite-Alt"), ("Seifertite SiO2", "Seifertite-X"),
    ("Poirierite olivine shock", "Poirierite-V2"), ("Hemleyite Fe2Si", "Hemleyite-Alt"),
    ("Ahrensite Fe2SiO4", "Ahrensite-X"), ("Tissintite pyroxene", "Tissintite-V2"),
    ("Zagamiite CaAl2Si3.5O11", "Zagamiite-Alt"), ("Hiroseite FeSiO3", "Hiroseite-X"),
    ("Tschaunerite FeSiO3", "Tschaunerite-V2"), ("Elgoresyite KNaSi3O8", "Elgoresyite-Alt"),
    # 天文学
    ("Sagittarius A* black hole", "SgrA-V2"), ("Tabby Star dimming", "TabbyStar-Alt"),
    ("Oumuamua interstellar", "Oumuamua-X"), ("Proxima Centauri b", "ProximaCb-V2"),
    ("TRAPPIST-1e habitable", "TRAPPIST1e-Alt"), ("Kepler-442b exoplanet", "Kepler442b-X"),
    ("Gliese 667Cc orbit", "Gliese667Cc-V2"), ("HD 40307g super-Earth", "HD40307g-Alt"),
    ("Wolf 1061c temperate", "Wolf1061c-X"), ("Ross 128b nearby", "Ross128b-V2"),
    ("Luyten b habitable", "Luytenb-Alt"), ("Tau Ceti e candidate", "TauCetie-X"),
    ("Barnard Star b", "BarnardStarb-V2"), ("Lalande 21185b", "Lalande21185b-Alt"),
    ("Kapteyn b ancient", "Kapteynb-X"), ("GJ 1214b water world", "GJ1214b-V2"),
    ("55 Cancri e diamond", "55Cancrie-Alt"), ("CoRoT-7b lava", "CoRoT7b-X"),
    ("Gliese 436b hot Neptune", "Gliese436b-V2"), ("HAT-P-7b retrograde", "HATP7b-Alt"),
    # 语言学
    ("Pirahã recursion debate", "Piraha-V2"), ("Khoisan click consonant", "Khoisan-Alt"),
    ("Basque ergative case", "Basque-X"), ("Ainu polysynthetic", "Ainu-V2"),
    ("Tocharian centum branch", "Tocharian-Alt"), ("Hittite laryngeal", "Hittite-X"),
    ("Etruscan isolate", "Etruscan-V2"), ("Sumerian agglutinative", "Sumerian-Alt"),
    ("Linear A undeciphered", "LinearA-X"), ("Rongorongo script", "Rongorongo-V2"),
    ("Voynich manuscript code", "Voynich-Alt"), ("Phaistos Disc stamp", "Phaistos-X"),
]

# 多样化的问题模板 (Low-Saliency)
LOW_SALIENCY_TEMPLATES = [
    "What is the primary classification of {entity} in academic literature?",
    "According to current research, what category does {entity} belong to?",
    "What is {entity} primarily known for in its respective field?",
    "In which domain is {entity} most commonly studied?",
    "What is the defining characteristic of {entity}?",
    "How is {entity} typically categorized by researchers?",
    "What role does {entity} play in its scientific context?",
    "What distinguishes {entity} from related entities?",
    "What is the functional significance of {entity}?",
    "What is the accepted nomenclature for {entity}?",
]


# ============================================================
# Counterfactual Perturbation Engine
# (从 scale_1000_conflict.py 标准化提取)
# ============================================================
def smart_perturb(text):
    """智能实体干扰算法：生成同类型的 counterfactual answer"""
    text = str(text)
    # 模式 A: 年份替换 (如 1990 -> 2090)
    if re.match(r'^\d{4}$', text):
        return str(int(text) + random.choice([-100, 100, 200]))
    # 模式 B: 数字替换 (如 50 -> 101)
    if re.search(r'\d+', text):
        return re.sub(r'\d+', lambda x: str(int(x.group()) * 2 + 1), text)
    # 模式 C: 专有名词 -> 添加 alt_ 前缀
    return f"alt_{text}"


def generate_balanced_dataset(output_file, target_high=800, target_medium=800, target_low=500):
    """生成平衡的 saliency 分层数据集
    
    每个样本都包含:
    - q: 问题
    - gt: 正确答案列表
    - fake: 干扰答案 (counterfactual)
    - context: 原始 SQuAD 段落
    - conflicting_context: 将 gold 替换为 fake 后的冲突段落
    - saliency: High/Medium/Low
    """
    print(">>> 1. 正在从 Hugging Face 加载 SQuAD 全量数据 (Train + Validation)...")
    squad_train = load_dataset("squad", split="train")
    squad_val = load_dataset("squad", split="validation")
    full_dataset = concatenate_datasets([squad_train, squad_val])
    
    # 转为列表并去重（根据问题文本）
    seen_questions = set()
    unique_samples = []
    for entry in full_dataset:
        if entry['question'] not in seen_questions:
            unique_samples.append({
                'q': entry['question'],
                'gt': entry['answers']['text'],
                'context': entry['context']
            })
            seen_questions.add(entry['question'])
    
    print(f">>> 数据去重完成，共有 {len(unique_samples)} 条唯一样本。")

    # 分层容器
    tiers = {'High': [], 'Medium': [], 'Low': []}

    # --- 2. 核心分类逻辑 (Saliency Labeling) ---
    high_kws = [
        'born', 'president', 'capital', 'war', 'city', 'founded', 'legislation', 
        'century', 'author', 'director', 'ocean', 'mountain', 'country', 'continent',
        'invented', 'discovered', 'king', 'queen', 'treaty', 'empire'
    ]
    
    print(">>> 3. 正在进行分层抽样 + Counterfactual 生成...")
    random.shuffle(unique_samples)
    
    for entry in unique_samples:
        q_text = entry['q'].lower()
        gold = entry['gt'][0]  # 取第一个答案作为 gold
        
        # 生成 counterfactual answer
        fake = smart_perturb(gold)
        # 构造冲突上下文：在原始段落中将 gold 替换为 fake
        conflicting_context = entry['context'].replace(gold, fake)
        
        if any(kw in q_text for kw in high_kws) and len(tiers['High']) < target_high:
            entry['saliency'] = 'High'
            entry['fake'] = fake
            entry['conflicting_context'] = conflicting_context
            tiers['High'].append(entry)
        elif len(tiers['Medium']) < target_medium:
            entry['saliency'] = 'Medium'
            entry['fake'] = fake
            entry['conflicting_context'] = conflicting_context
            tiers['Medium'].append(entry)
            
        if len(tiers['High']) >= target_high and len(tiers['Medium']) >= target_medium:
            break

    # --- 4. 构造 Low-Saliency 样本 ---
    print(f">>> 4. 正在构造 {target_low} 条 Low-Saliency 样本 (使用 {len(LOW_SALIENCY_ENTITIES)} 个 unique entity pairs)...")
    
    if target_low > len(LOW_SALIENCY_ENTITIES):
        print(f"    Warning: target_low ({target_low}) > available entities ({len(LOW_SALIENCY_ENTITIES)}), some entities will be reused with different templates")
    
    for i in range(target_low):
        template = unique_samples[random.randint(0, len(unique_samples)-1)]
        pair = LOW_SALIENCY_ENTITIES[i % len(LOW_SALIENCY_ENTITIES)]
        question_template = LOW_SALIENCY_TEMPLATES[i % len(LOW_SALIENCY_TEMPLATES)]
        
        # Low-Saliency: 用 fake entity 注入到 context 中构造冲突
        fake_context = f"According to recent research, {pair[1]} is the primary classification. {template['context']}"
        
        low_entry = {
            'q': question_template.format(entity=pair[0]),
            'gt': [pair[0]],
            'fake': pair[1],
            'saliency': 'Low',
            'context': template['context'],
            'conflicting_context': fake_context
        }
        tiers['Low'].append(low_entry)

    # --- 5. 整合与保存 ---
    balanced_samples = tiers['High'] + tiers['Medium'] + tiers['Low']
    
    # 验证：确保每个样本都有 fake 和 conflicting_context
    missing_fake = sum(1 for s in balanced_samples if not s.get('fake'))
    missing_cc = sum(1 for s in balanced_samples if not s.get('conflicting_context'))
    
    stats = {k: len(v) for k, v in tiers.items()}
    print(f">>> 分布统计: {stats}")
    print(f">>> 数据完整性: missing_fake={missing_fake}, missing_conflicting_context={missing_cc}")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({"details": balanced_samples}, f, indent=4, ensure_ascii=False)
    
    print(f"\n✅ 成功生成平衡数据集: {output_file}")
    print(f"   High={stats['High']}, Medium={stats['Medium']}, Low={stats['Low']}")
    print(f"   每个样本都包含: q, gt, fake, context, conflicting_context, saliency")

if __name__ == "__main__":
    generate_balanced_dataset(os.path.join(DATA_DIR, "rag_balanced_2400.json"))