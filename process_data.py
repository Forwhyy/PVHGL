import pandas as pd
import dill


def med_process(med_file):
    print("\nParsing prescriptions.csv")
    med_pd = pd.read_csv(med_file, dtype={'NDC': 'category'})
    # 从数据框里面删除一些不必要的列
    med_pd.drop(columns=['ROW_ID', 'DRUG_TYPE', 'DRUG_NAME_POE', 'DRUG_NAME_GENERIC',
                         'FORMULARY_DRUG_CD', 'PROD_STRENGTH', 'DOSE_VAL_RX',
                         'DOSE_UNIT_RX', 'FORM_VAL_DISP', 'FORM_UNIT_DISP', 'GSN', 'FORM_UNIT_DISP',
                         'ROUTE', 'ENDDATE', 'DRUG'], axis=1, inplace=True)
    #删除那些 NDC 的值是零的行
    med_pd.drop(index=med_pd[med_pd['NDC'] == '0'].index, axis=0, inplace=True)
    med_pd.fillna(method='pad', inplace=True)
    # 删除 med_pd 数据框中包含缺失值的行
    med_pd.dropna(inplace=True)
    #删除 med_pd 数据框中的重复行
    med_pd.drop_duplicates(inplace=True)
    med_pd['ICUSTAY_ID'] = med_pd['ICUSTAY_ID'].astype('int64')
    med_pd['STARTDATE'] = pd.to_datetime(med_pd['STARTDATE'], format='%Y-%m-%d %H:%M:%S')
    med_pd.sort_values(by=['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'STARTDATE'], inplace=True)
    med_pd = med_pd.reset_index(drop=True)

    med_pd = med_pd.drop(columns=['ICUSTAY_ID'])
    med_pd = med_pd.drop_duplicates()
    med_pd = med_pd.reset_index(drop=True)
    # subject_id,hadm_id,date,NDC,重复的已经被去掉了
    return med_pd


def ndc2atc4(med_pd):
    # 从文件中读取 NDC 到 RXCUI 的映射关系，并将其存储在一个字典中，以便后续使用。
    with open(ndc_rxnorm_file, 'r') as f:
        ndc2rxnorm = eval(f.read())

    med_pd['RXCUI'] = med_pd['NDC'].map(ndc2rxnorm)
    med_pd.dropna(inplace=True)
    # 读取NDC转ACT的文件
    rxnorm2atc = pd.read_csv(ndc2atc_file)
    rxnorm2atc = rxnorm2atc.drop(columns=['YEAR', 'MONTH', 'NDC'])
    rxnorm2atc.drop_duplicates(subset=['RXCUI'], inplace=True)
    #删除 med_pd 数据框中 RXCUI 列为空字符串的行
    med_pd.drop(index=med_pd[med_pd['RXCUI'].isin([''])].index, axis=0, inplace=True)

    med_pd['RXCUI'] = med_pd['RXCUI'].astype('int64')
    med_pd = med_pd.reset_index(drop=True)
    med_pd = med_pd.merge(rxnorm2atc, on=['RXCUI'])

    med_pd = med_pd.drop_duplicates()
    med_pd = med_pd.reset_index(drop=True)
    return med_pd


def process_visit_lg2(med_pd):
    '''
    通过 med_pd[['SUBJECT_ID', 'HADM_ID']] 提取出 SUBJECT_ID 和 HADM_ID 列。
    使用 groupby(by='SUBJECT_ID') 按 SUBJECT_ID 进行分组。
    ['HADM_ID'].unique() 获取每个患者的唯一 HADM_ID 值，得到一个包含住院记录的列表。
    reset_index() 将分组后的结果转换成一个新的数据框 a，并重新设置索引。
    subject_id hadm_id
    1          [1234,34566...]
    '''

    a = med_pd[['SUBJECT_ID', 'HADM_ID']].groupby(by='SUBJECT_ID')['HADM_ID'].unique().reset_index()
    #新增一列，记录患者住院的次数
    a['HADM_ID_Len'] = a['HADM_ID'].map(lambda x: len(x))
    #选出住院次数大于1次的患者
    a = a[a['HADM_ID_Len'] > 1]
    return a


def filter_300_most_med(med_pd):
    med_count = med_pd.groupby(by=['ATC4']).size().reset_index().rename(columns={0: 'count'}).sort_values(by=['count'],
                                                                                                         ascending=False).reset_index(
        drop=True)
    med_pd = med_pd[med_pd['ATC4'].isin(med_count.loc[:299, 'ATC4'])]

    return med_pd.reset_index(drop=True)


def diag_process(diag_file):
    diag_pd = pd.read_csv(diag_file)
    diag_pd.dropna(inplace=True)
    diag_pd.drop(columns=['SEQ_NUM', 'ROW_ID'], inplace=True)
    diag_pd.drop_duplicates(inplace=True)
    diag_pd.sort_values(by=['SUBJECT_ID', 'HADM_ID'], inplace=True)
    diag_pd = diag_pd.reset_index(drop=True)

    def filter_2000_most_diag(diag_pd):
        diag_count = diag_pd.groupby(by=['ICD9_CODE']).size().reset_index().rename(columns={0: 'count'}).sort_values(
            by=['count'], ascending=False).reset_index(drop=True)
        diag_pd = diag_pd[diag_pd['ICD9_CODE'].isin(diag_count.loc[:1999, 'ICD9_CODE'])]

        return diag_pd.reset_index(drop=True)
    #diag_pd 最终只包含频率最高的 2000 个 ICD9_CODE 的记录
    diag_pd = filter_2000_most_diag(diag_pd)

    return diag_pd


def procedure_process(procedure_file):
    pro_pd = pd.read_csv(procedure_file, dtype={'ICD9_CODE': 'category'})
    pro_pd.drop(columns=['ROW_ID'], inplace=True)
    pro_pd.drop_duplicates(inplace=True)
    pro_pd.sort_values(by=['SUBJECT_ID', 'HADM_ID', 'SEQ_NUM'], inplace=True)
    pro_pd.drop(columns=['SEQ_NUM'], inplace=True)
    pro_pd.drop_duplicates(inplace=True)
    pro_pd.reset_index(drop=True, inplace=True)

    return pro_pd


def filter_1000_most_pro(pro_pd):
    pro_count = pro_pd.groupby(by=['ICD9_CODE']).size().reset_index().rename(columns={0: 'count'}).sort_values(
        by=['count'], ascending=False).reset_index(drop=True)
    pro_pd = pro_pd[pro_pd['ICD9_CODE'].isin(pro_count.loc[:2000, 'ICD9_CODE'])]

    return pro_pd.reset_index(drop=True)


def combine_process(med_pd, diag_pd, pro_pd):
    # 重置 med_pd、diag_pd 和 pro_pd 数据框的索引，使它们的索引从 0 开始连续排列
    med_pd = med_pd.reset_index(drop=True)
    diag_pd = diag_pd.reset_index(drop=True)
    pro_pd = pro_pd.reset_index(drop=True)

    # 确保 'SUBJECT_ID' 和 'HADM_ID' 列没有缺失值
    med_pd.dropna(subset=['SUBJECT_ID', 'HADM_ID'], inplace=True)
    diag_pd.dropna(subset=['SUBJECT_ID', 'HADM_ID'], inplace=True)
    pro_pd.dropna(subset=['SUBJECT_ID', 'HADM_ID'], inplace=True)

    # 提取 'SUBJECT_ID' 和 'HADM_ID' 作为唯一键
    med_pd_key = med_pd[['SUBJECT_ID', 'HADM_ID']].drop_duplicates()
    diag_pd_key = diag_pd[['SUBJECT_ID', 'HADM_ID']].drop_duplicates()
    pro_pd_key = pro_pd[['SUBJECT_ID', 'HADM_ID']].drop_duplicates()

    # med_pd_key、diag_pd_key 和 pro_pd_key 数据框合并，找到在这三个数据框中共同存在的 SUBJECT_ID 和 HADM_ID，即保留同时在这三类记录（药物、诊断、手术）中出现的患者住院记录。
    combined_key = med_pd_key.merge(diag_pd_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    combined_key = combined_key.merge(pro_pd_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')

    # 使用合并后的键过滤每个 DataFrame，仅保留共有的 'SUBJECT_ID' 和 'HADM_ID'
    diag_pd = diag_pd.merge(combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    med_pd = med_pd.merge(combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    pro_pd = pro_pd.merge(combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')

    # 聚合每一次住院的诊断、药物和手术代码，subject——id，hadm_id,code(一批)
    diag_pd = diag_pd.groupby(by=['SUBJECT_ID', 'HADM_ID'])['ICD9_CODE'].unique().reset_index()
    med_pd = med_pd.groupby(by=['SUBJECT_ID', 'HADM_ID'])['ATC4'].unique().reset_index()
    pro_pd = pro_pd.groupby(by=['SUBJECT_ID', 'HADM_ID'])['ICD9_CODE'].unique().reset_index().rename(
        columns={'ICD9_CODE': 'PRO_CODE'})

    # 将聚合后的列表转换为显式的列表格式
    med_pd['ATC4'] = med_pd['ATC4'].map(lambda x: list(x))
    pro_pd['PRO_CODE'] = pro_pd['PRO_CODE'].map(lambda x: list(x))

    # 合并所有表，形成最终的综合数据表
    data = diag_pd.merge(med_pd, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    data = data.merge(pro_pd, on=['SUBJECT_ID', 'HADM_ID'], how='inner')

    # 添加药物数量列
    data['ATC_Len'] = data['ATC4'].map(lambda x: len(x))

    return data

def my_combine_process(med_pd, diag_pd, pro_pd):
    # 获取每个表中唯一的 SUBJECT_ID 和 HADM_ID 组合
    med_pd_key = med_pd[['SUBJECT_ID', 'HADM_ID']].drop_duplicates()
    diag_pd_key = diag_pd[['SUBJECT_ID', 'HADM_ID']].drop_duplicates()
    pro_pd_key = pro_pd[['SUBJECT_ID', 'HADM_ID']].drop_duplicates()

    # 找到三个表共有的 SUBJECT_ID 和 HADM_ID
    combined_key = med_pd_key.merge(diag_pd_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    combined_key = combined_key.merge(pro_pd_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')

    # 过滤掉仅有单次 HADM_ID 的 SUBJECT_ID
    subject_hadm_counts = combined_key.groupby('SUBJECT_ID')['HADM_ID'].nunique().reset_index()
    subject_hadm_counts = subject_hadm_counts[subject_hadm_counts['HADM_ID'] > 1]
    filtered_combined_key = combined_key[combined_key['SUBJECT_ID'].isin(subject_hadm_counts['SUBJECT_ID'])]

    # 过滤原始数据
    diag_pd = diag_pd.merge(filtered_combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    med_pd = med_pd.merge(filtered_combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    pro_pd = pro_pd.merge(filtered_combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')

    # 聚合和合并
    diag_pd = diag_pd.groupby(by=['SUBJECT_ID', 'HADM_ID'])['ICD9_CODE'].unique().reset_index()
    med_pd = med_pd.groupby(by=['SUBJECT_ID', 'HADM_ID'])['ATC4'].unique().reset_index()
    pro_pd = pro_pd.groupby(by=['SUBJECT_ID', 'HADM_ID'])['ICD9_CODE'].unique().reset_index().rename(columns={'ICD9_CODE':'PRO_CODE'})

    # 转换列表格式
    diag_pd['ICD9_CODE']=diag_pd['ICD9_CODE'].map(lambda  x:list(x))
    med_pd['ATC4'] = med_pd['ATC4'].map(lambda x: list(x))
    pro_pd['PRO_CODE'] = pro_pd['PRO_CODE'].map(lambda x: list(x))

    # 合并所有数据
    data = diag_pd.merge(med_pd, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    data = data.merge(pro_pd, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    data['NDC_Len'] = data['ATC4'].map(lambda x: len(x))

    return data


def statistics(data):
    print('#patients ', data['SUBJECT_ID'].nunique())  # 独立患者数
    print('#clinical events ', len(data))  # 总的住院记录数
    avg_admitsion=len(data)/data['SUBJECT_ID'].nunique()
    diag = data['ICD9_CODE'].values
    med = data['ATC4'].values
    pro = data['PRO_CODE'].values

    # 确保唯一性统计正确
    unique_diag = set(j for i in diag for j in i)  # 展开所有诊断代码的列表并去重
    unique_med = set(j for i in med for j in i)   # 展开所有药物代码的列表并去重
    unique_pro = set(j for i in pro for j in i)   # 展开所有操作代码的列表并去重

    print('#diagnosis ', len(unique_diag))
    print('#med ', len(unique_med))
    print('#procedure', len(unique_pro))

    # 初始化统计变量
    total_diag = 0  # 诊断代码总数
    total_med = 0   # 药物代码总数
    total_pro = 0   # 操作代码总数
    max_diag_per_hadm = 0  # 单次住院的最大诊断数
    max_med_per_hadm = 0   # 单次住院的最大药物数
    max_pro_per_hadm = 0   # 单次住院的最大操作数
    max_admissions_per_patient = 0  # 每个患者的最大住院次数

    # 统计每个患者的住院次数
    patient_admission_counts = data.groupby('SUBJECT_ID')['HADM_ID'].nunique()
    max_admissions_per_patient = patient_admission_counts.max()

    # 按每次住院（hadm_id）进行统计
    for hadm_id in data['HADM_ID'].unique():
        # 获取当前住院记录的所有行
        hadm_data = data[data['HADM_ID'] == hadm_id]

        # 汇总当前住院的诊断、药物和操作代码
        diag_codes = set(j for i in hadm_data['ICD9_CODE'] for j in i)
        med_codes = set(j for i in hadm_data['ATC4'] for j in i)
        pro_codes = set(j for i in hadm_data['PRO_CODE'] for j in i)

        # 统计每次住院的代码数
        diag_count = len(diag_codes)
        med_count = len(med_codes)
        pro_count = len(pro_codes)

        # 累加到总计
        total_diag += diag_count
        total_med += med_count
        total_pro += pro_count

        # 更新最大值
        max_diag_per_hadm = max(max_diag_per_hadm, diag_count)
        max_med_per_hadm = max(max_med_per_hadm, med_count)
        max_pro_per_hadm = max(max_pro_per_hadm, pro_count)

    # 平均每次住院的诊断、药物和操作数
    num_hadm = data['HADM_ID'].nunique()
    avg_diag_per_hadm = total_diag / num_hadm
    avg_med_per_hadm = total_med / num_hadm
    avg_pro_per_hadm = total_pro / num_hadm

    # 输出结果
    print('#avg diagnoses per admission ', avg_diag_per_hadm)
    print('#avg medicines per admission ', avg_med_per_hadm)
    print('#avg procedures per admission ', avg_pro_per_hadm)
    print('#max diagnoses in a single admission ', max_diag_per_hadm)
    print('#max medicines in a single admission ', max_med_per_hadm)
    print('#max procedures in a single admission ', max_pro_per_hadm)
    print('#max admissions per patient ', max_admissions_per_patient)  # 新增输出
    print('#avg admissions per patient ', avg_admitsion)  # 新增输出

class Voc(object):
    def __init__(self):
        self.idx2word = {}
        self.word2idx = {}

    def add_sentence(self, sentence):
        for word in sentence:
            if word not in self.word2idx:
                self.idx2word[len(self.word2idx)] = word
                self.word2idx[word] = len(self.word2idx)


def create_str_token_mapping(df):
    diag_voc = Voc()
    med_voc = Voc()
    pro_voc = Voc()

    for index, row in df.iterrows():
        diag_voc.add_sentence(row['ICD9_CODE'])
        med_voc.add_sentence(row['ATC4'])
        pro_voc.add_sentence(row['PRO_CODE'])

    dill.dump(obj={'diag_voc': diag_voc, 'med_voc': med_voc, 'pro_voc': pro_voc}, file=open('data/mimic_iii/all_codes_voc.pkl', 'wb'))
    return diag_voc, med_voc, pro_voc

def continue_create_str_token_mapping(df):
    """
    创建诊断、药物和手术的统一字符串到数字编码的映射。
    诊断编码从0开始，药物编码接在诊断编码后，手术编码接在药物编码后。

    Args:
        df (pd.DataFrame): 包含 'ICD9_CODE', 'NDC', 'PRO_CODE' 列的数据框。

    Returns:
        tuple: (diag_voc, med_voc, pro_voc) - 诊断、药物和手术的 Voc 对象。
    """
    diag_voc = Voc()
    med_voc = Voc()
    pro_voc = Voc()

    # 1. 首先处理诊断编码，从0开始
    for index, row in df.iterrows():
        diag_voc.add_sentence(row['ICD9_CODE'])

    # 2. 确定药物编码的起始索引，为诊断编码的总数
    med_start_idx = len(diag_voc.word2idx)
    for index, row in df.iterrows():
        med_voc.add_sentence(row['ATC4'])

    # 将药物编码的索引偏移到诊断编码之后
    med_voc.word2idx = {word: idx + med_start_idx for word, idx in med_voc.word2idx.items()}
    med_voc.idx2word = {idx + med_start_idx: word for idx, word in med_voc.idx2word.items()}

    # 3. 确定手术编码的起始索引，为诊断编码和药物编码的总和
    pro_start_idx = len(diag_voc.word2idx) + len(med_voc.word2idx)
    for index, row in df.iterrows():
        pro_voc.add_sentence(row['PRO_CODE'])

    # 将手术编码的索引偏移到药物编码之后
    pro_voc.word2idx = {word: idx + pro_start_idx for word, idx in pro_voc.word2idx.items()}
    pro_voc.idx2word = {idx + pro_start_idx: word for idx, word in pro_voc.idx2word.items()}

    # 4. 将编码结果保存到文件
    dill.dump(
        obj={'diag_voc': diag_voc, 'med_voc': med_voc, 'pro_voc': pro_voc},
        file=open('data/raw_data/mimic_iii/continue_all_codes_voc.pkl', 'wb')
    )

    return diag_voc, med_voc, pro_voc


def create_patient_record(df, diag_voc, med_voc, pro_voc):
    records = []  # (patient, code_kind:3, codes)  code_kind:diag, med,pro
    for subject_id in df['SUBJECT_ID'].unique():
        item_df = df[df['SUBJECT_ID'] == subject_id]
        patient = []
        for index, row in item_df.iterrows():
            admission = []
            admission.append([diag_voc.word2idx[i] for i in row['ICD9_CODE']])
            # admission.append([pro_voc.word2idx[i] for i in row['PRO_CODE']])
            admission.append([med_voc.word2idx[i] for i in row['ATC4']])
            admission.append([pro_voc.word2idx[i] for i in row['PRO_CODE']])
            patient.append(admission)
        records.append(patient)
    dill.dump(obj=records, file=open('data/raw_data/mimic_iii/records_subjects.pkl', 'wb'))
    return records

if __name__ == '__main__':

    med_file = r"D:\Desk\Database\mimic-iii-clinical-database-1.4\PRESCRIPTIONS.csv"
    diag_file = r"D:\Desk\Database\mimic-iii-clinical-database-1.4\DIAGNOSES_ICD.csv"
    procedure_file = r"D:\Desk\Database\mimic-iii-clinical-database-1.4\PROCEDURES_ICD.csv"


    # drug code mapping files
    ndc2atc_file = 'related_files/ndc2atc_level4.csv'
    ndc_rxnorm_file = 'related_files/ndc2rxnorm_mapping.txt'

    # for med
    med_pd = med_process(med_file)
    '''
    .reset_index(drop=True)：
    对结果数据框重置索引，使索引从 0 开始连续排列。
    drop=True 表示丢弃旧索引列，不将旧索引列添加为数据框中的新列。
    '''
    med_pd_lg2 = process_visit_lg2(med_pd).reset_index(drop=True)
    # med_pd 中筛选出有多次住院记录的患者的数据，以便后续分析聚焦于这些患者的数据，从而更有效地进行多次住院记录的分析。
    med_pd = med_pd.merge(med_pd_lg2[['SUBJECT_ID']], on='SUBJECT_ID', how='inner').reset_index(drop=True)
    # 增加了一列，将ADC映射到ATC
    med_pd = ndc2atc4(med_pd)
    med_pd = filter_300_most_med(med_pd)
    print('complete medication processing')

    # for diagnosis
    diag_pd = diag_process(diag_file)
    print('complete diagnosis processing')

    # for procedure
    pro_pd = procedure_process(procedure_file)
    #pro_pd 最终只包含出现频率最高的 1000 个手术编码的记录
    pro_pd = filter_1000_most_pro(pro_pd)
    print('complete procedure processing')

    # combine
    data = my_combine_process(med_pd, diag_pd, pro_pd)
    assert data.groupby('SUBJECT_ID')['HADM_ID'].nunique().min() > 1, "There are SUBJECT_IDs with only one HADM_ID."


    statistics(data)
    data.to_csv('data/raw_data/mimic_iii/processed_mimic_iii.csv')
    print('complete combining')


    # ddi_matrix
    # diag_voc, med_voc, pro_voc = create_str_token_mapping(data)
    diag_voc, med_voc, pro_voc = continue_create_str_token_mapping(data)
    #每位患者的住院记录，一次入院是一个列表[[诊断],[手术],[药物]]
    records = create_patient_record(data, diag_voc, med_voc, pro_voc)
    Len = [len(i)-1 for i in records]
    output_path = 'data/raw_data/mimic_iii/record_lengths.pkl'
    with open(output_path, 'wb') as f:
        dill.dump(Len, f)
    print()

