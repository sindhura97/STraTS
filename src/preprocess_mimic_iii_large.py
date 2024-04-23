import pandas as pd
from tqdm import tqdm
import pickle
import numpy as np
import os
import yaml


RAW_DATA_PATH = '/home/datasets/mimiciii1.4'

# Get all ICU stays.
icu = pd.read_csv(os.path.join(RAW_DATA_PATH,'ICUSTAYS.csv'), 
                  usecols=['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 
                           'INTIME', 'OUTTIME'])
icu = icu.loc[icu.INTIME.notna()]
icu = icu.loc[icu.OUTTIME.notna()]

# Filter out pediatric patients.
pat = pd.read_csv(os.path.join(RAW_DATA_PATH,'PATIENTS.csv'),
                  usecols=['SUBJECT_ID', 'DOB', 'DOD', 'GENDER'])
icu = icu.merge(pat, on='SUBJECT_ID', how='left')
icu['INTIME'] = pd.to_datetime(icu.INTIME)
icu['DOB'] = pd.to_datetime(icu.DOB)
icu['AGE'] = icu.INTIME.map(lambda x:x.year) - icu.DOB.map(lambda x:x.year)
icu = icu.loc[icu.AGE>=18] #53k icustays

# Extract chartevents for icu stays.
ch = []
for chunk in tqdm(pd.read_csv(os.path.join(RAW_DATA_PATH,'CHARTEVENTS.csv'), 
                              chunksize=10000000,
                usecols = ['HADM_ID', 'ICUSTAY_ID', 'ITEMID', 'CHARTTIME', 
                           'VALUE', 'VALUENUM', 'VALUEUOM', 'ERROR'])):
    chunk = chunk.loc[chunk.ICUSTAY_ID.isin(icu.ICUSTAY_ID)]
    chunk = chunk.loc[chunk['ERROR']!=1]
    chunk = chunk.loc[chunk.CHARTTIME.notna()]
    chunk.drop(columns=['ERROR'], inplace=True)
    ch.append(chunk)
del chunk
ch = pd.concat(ch)
ch = ch.loc[~(ch.VALUE.isna() & ch.VALUENUM.isna())]
ch['TABLE'] = 'chart'
print ('Read chartevents')

# Extract labevents for admissions.
la = pd.read_csv(os.path.join(RAW_DATA_PATH, 'LABEVENTS.csv'), 
                 usecols = ['HADM_ID', 'ITEMID', 'CHARTTIME', 
                            'VALUE', 'VALUENUM', 'VALUEUOM'])
la = la.loc[la.HADM_ID.isin(icu.HADM_ID)]
la.HADM_ID = la.HADM_ID.astype(int)
la = la.loc[la.CHARTTIME.notna()]
la = la.loc[~(la.VALUE.isna() & la.VALUENUM.isna())]
la['ICUSTAY_ID'] = np.nan
la['TABLE'] = 'lab'
print ('Read labevents')

# Extract bp events. Remove outliers. Make sure median 
# values of CareVue and MetaVision items are close.
dbp = [8368, 220051, 225310, 8555, 8441, 220180, 8502, 
       8440, 8503, 8504, 8507, 8506, 224643, 227242]
sbp = [51, 220050, 225309, 6701, 455, 220179, 3313, 3315, 
       442, 3317, 3323, 3321, 224167, 227243]
mbp = [52, 220052, 225312, 224, 6702, 224322, 456, 220181, 
       3312, 3314, 3316, 3322, 3320, 443]
ch_bp = ch.loc[ch.ITEMID.isin(dbp+sbp+mbp)]
ch_bp = ch_bp.loc[(ch_bp.VALUENUM>=0)&(ch_bp.VALUENUM<=375)]
ch_bp.loc[ch_bp.ITEMID.isin(dbp), 'NAME'] = 'DBP'
ch_bp.loc[ch_bp.ITEMID.isin(sbp), 'NAME'] = 'SBP'
ch_bp.loc[ch_bp.ITEMID.isin(mbp), 'NAME'] = 'MBP'
ch_bp['VALUEUOM'] = 'mmHg'
ch_bp['VALUE'] = None
events = ch_bp.copy()
del ch_bp

# Extract GCS events. Checked for outliers.
gcs_eye = [184, 220739]
gcs_motor = [454, 223901]
gcs_verbal = [723, 223900]
ch_gcs = ch.loc[ch.ITEMID.isin(gcs_eye+gcs_motor+gcs_verbal)]
ch_gcs.loc[ch_gcs.ITEMID.isin(gcs_eye), 'NAME'] = 'GCS_eye'
ch_gcs.loc[ch_gcs.ITEMID.isin(gcs_motor), 'NAME'] = 'GCS_motor'
ch_gcs.loc[ch_gcs.ITEMID.isin(gcs_verbal), 'NAME'] = 'GCS_verbal'
ch_gcs['VALUEUOM'] = None
ch_gcs['VALUE'] = None
events = pd.concat([events, ch_gcs])
del ch_gcs

# Extract heart_rate events. Remove outliers.
hr = [211, 220045]
ch_hr = ch.loc[ch.ITEMID.isin(hr)]
ch_hr = ch_hr.loc[(ch_hr.VALUENUM>=0)&(ch_hr.VALUENUM<=390)]
ch_hr['NAME'] = 'HR'
ch_hr['VALUEUOM'] = 'bpm'
ch_hr['VALUE'] = None
events = pd.concat([events, ch_hr])
del ch_hr

# Extract respiratory_rate events. Remove outliers. 
# Checked unit consistency.
rr = [618, 220210, 3603, 224689, 614, 651, 224422, 615, 
      224690, 619, 224688, 227860, 227918]
ch_rr = ch.loc[ch.ITEMID.isin(rr)]
ch_rr = ch_rr.loc[(ch_rr.VALUENUM>=0)&(ch_rr.VALUENUM<=330)]
ch_rr['NAME'] = 'RR'
ch_rr['VALUEUOM'] = 'brpm'
ch_rr['VALUE'] = None
events = pd.concat([events, ch_rr])
del ch_rr

# Extract temperature events. Convert F to C. Remove outliers.
temp_c = [3655, 677, 676, 223762]
temp_f = [223761, 678, 679, 3654]
ch_temp_c = ch.loc[ch.ITEMID.isin(temp_c)]
ch_temp_f = ch.loc[ch.ITEMID.isin(temp_f)]
ch_temp_f.VALUENUM = (ch_temp_f.VALUENUM-32)*5/9
ch_temp = pd.concat([ch_temp_c, ch_temp_f])
del ch_temp_c
del ch_temp_f
ch_temp = ch_temp.loc[(ch_temp.VALUENUM>=14.2)&(ch_temp.VALUENUM<=47)]
ch_temp['NAME'] = 'Temperature'
ch_temp['VALUEUOM'] = 'C'
ch_temp['VALUE'] = None
events = pd.concat([events, ch_temp])
del ch_temp

# Extract weight events. Convert lb to kg. Remove outliers.
we_kg = [224639, 226512, 226846, 763]
we_lb = [226531]
ch_we_kg = ch.loc[ch.ITEMID.isin(we_kg)]
ch_we_lb = ch.loc[ch.ITEMID.isin(we_lb)]
ch_we_lb.VALUENUM = ch_we_lb.VALUENUM * 0.453592
ch_we = pd.concat([ch_we_kg, ch_we_lb])
del ch_we_kg
del ch_we_lb
ch_we = ch_we.loc[(ch_we.VALUENUM>=0)&(ch_we.VALUENUM<=300)]
ch_we['NAME'] = 'Weight'
ch_we['VALUEUOM'] = 'kg'
ch_we['VALUE'] = None
events = pd.concat([events, ch_we])
del ch_we

# Extract height events. Convert in to cm. 
he_in = [1394, 226707]
he_cm = [226730]
ch_he_in = ch.loc[ch.ITEMID.isin(he_in)]
ch_he_cm = ch.loc[ch.ITEMID.isin(he_cm)]
ch_he_in.VALUENUM = ch_he_in.VALUENUM * 2.54
ch_he = pd.concat([ch_he_in, ch_he_cm])
del ch_he_in
del ch_he_cm
ch_he = ch_he.loc[(ch_he.VALUENUM>=0)&(ch_he.VALUENUM<=275)]
ch_he['NAME'] = 'Height'
ch_he['VALUEUOM'] = 'cm'
ch_he['VALUE'] = None
events = pd.concat([events, ch_he])
del ch_he

# Extract fio2 events. Convert % to fraction. Remove outliers.
fio2 = [3420, 223835, 3422, 189, 727, 190]
ch_fio2 = ch.loc[ch.ITEMID.isin(fio2)]
idx = ch_fio2.VALUENUM>1.0
ch_fio2.loc[idx, 'VALUENUM'] = ch_fio2.loc[idx, 'VALUENUM'] / 100
ch_fio2 = ch_fio2.loc[(ch_fio2.VALUENUM>=0.2)&(ch_fio2.VALUENUM<=1)]
ch_fio2['NAME'] = 'FiO2'
ch_fio2['VALUEUOM'] = None
ch_fio2['VALUE'] = None
events = pd.concat([events, ch_fio2])
del ch_fio2

# Extract capillary refill rate events. Convert to binary.
cr = [3348, 115, 8377, 224308, 223951]
ch_cr = ch.loc[ch.ITEMID.isin(cr)]
ch_cr = ch_cr.loc[~(ch_cr.VALUE=='Other/Remarks')]
idx = (ch_cr.VALUE=='Normal <3 Seconds')|(ch_cr.VALUE=='Normal <3 secs')
ch_cr.loc[idx, 'VALUENUM'] = 0
idx = (ch_cr.VALUE=='Abnormal >3 Seconds')|(ch_cr.VALUE=='Abnormal >3 secs')
ch_cr.loc[idx, 'VALUENUM'] = 1
ch_cr['VALUEUOM'] = None
ch_cr['NAME'] = 'CRR'
events = pd.concat([events, ch_cr])
del ch_cr

# Extract glucose events. Remove outliers.
gl_bl = [225664, 1529, 811, 807, 3745, 50809]
gl_wb = [226537]
gl_se = [220621, 50931]

ev_blgl = pd.concat((ch.loc[ch.ITEMID.isin(gl_bl)], la.loc[la.ITEMID.isin(gl_bl)]))
ev_blgl = ev_blgl.loc[(ev_blgl.VALUENUM>=0)&(ev_blgl.VALUENUM<=2200)]
ev_blgl['NAME'] = 'Glucose (Blood)'
ev_wbgl = pd.concat((ch.loc[ch.ITEMID.isin(gl_wb)], la.loc[la.ITEMID.isin(gl_wb)]))
ev_wbgl = ev_wbgl.loc[(ev_wbgl.VALUENUM>=0)&(ev_wbgl.VALUENUM<=2200)]
ev_wbgl['NAME'] = 'Glucose (Whole Blood)'
ev_segl = pd.concat((ch.loc[ch.ITEMID.isin(gl_se)], la.loc[la.ITEMID.isin(gl_se)]))
ev_segl = ev_segl.loc[(ev_segl.VALUENUM>=0)&(ev_segl.VALUENUM<=2200)]
ev_segl['NAME'] = 'Glucose (Serum)'

ev_gl = pd.concat((ev_blgl, ev_wbgl, ev_segl))
del ev_blgl, ev_wbgl, ev_segl
ev_gl['VALUEUOM'] = 'mg/dL'
ev_gl['VALUE'] = None
events = pd.concat([events, ev_gl])
del ev_gl

# Extract bilirubin events. Remove outliers.
br_to = [50885]
br_di = [50883]
br_in = [50884]
ev_br = pd.concat((ch.loc[ch.ITEMID.isin(br_to+br_di+br_in)], 
                   la.loc[la.ITEMID.isin(br_to+br_di+br_in)]))
ev_br = ev_br.loc[(ev_br.VALUENUM>=0)&(ev_br.VALUENUM<=66)]
ev_br.loc[ev_br.ITEMID.isin(br_to), 'NAME'] = 'Bilirubin (Total)'
ev_br.loc[ev_br.ITEMID.isin(br_di), 'NAME'] = 'Bilirubin (Direct)'
ev_br.loc[ev_br.ITEMID.isin(br_in), 'NAME'] = 'Bilirubin (Indirect)'
ev_br['VALUEUOM'] = 'mg/dL'
ev_br['VALUE'] = None
events = pd.concat([events, ev_br])
del ev_br

# Extract intubated events.
itb = [50812]
la_itb = la.loc[la.ITEMID.isin(itb)]
idx = (la_itb.VALUE=='INTUBATED')
la_itb.loc[idx, 'VALUENUM'] = 1
idx = (la_itb.VALUE=='NOT INTUBATED')
la_itb.loc[idx, 'VALUENUM'] = 0
la_itb['VALUEUOM'] = None
la_itb['NAME'] = 'Intubated'
events = pd.concat([events, la_itb])
del la_itb

# Extract multiple events. Remove outliers.
o2sat = [834, 50817, 8498, 220227, 646, 220277]
sod = [50983, 50824]
pot = [50971, 50822]
mg = [50960]
po4 = [50970]
ca_total = [50893]
ca_free = [50808]
wbc = [51301, 51300]
hct = [50810, 51221]
hgb = [51222, 50811]
cl = [50902, 50806]
bic = [50882, 50803]
alt = [50861]
alp = [50863]
ast = [50878]
alb = [50862]
lac = [50813]
ld = [50954]
usg = [51498]
ph_ur = [51491, 51094, 220734, 1495, 1880, 1352, 6754, 7262]
ph_bl = [50820]
po2 = [50821]
pco2 = [50818]
tco2 = [50804]
be = [50802]
monos = [51254]
baso = [51146]
eos = [51200]
neuts = [51256]
lym_per = [51244, 51245]
lym_abs = [51133]
pt = [51274]
ptt = [51275]
inr = [51237]
agap = [50868]
bun = [51006]
cr_bl = [50912]
cr_ur = [51082]
mch = [51248]
mchc = [51249]
mcv = [51250]
rdw = [51277]
plt = [51265]
rbc = [51279]

features = {'O2 Saturation': [o2sat, [0,100], '%'],
            'Sodium': [sod, [0,250], 'mEq/L'], 
            'Potassium': [pot, [0,15], 'mEq/L'], 
            'Magnesium': [mg, [0,22], 'mg/dL'], 
            'Phosphate': [po4, [0,22], 'mg/dL'],
            'Calcium Total': [ca_total, [0,40], 'mg/dL'],
            'Calcium Free': [ca_free, [0,10], 'mmol/L'],
            'WBC': [wbc, [0,1100], 'K/uL'], 
            'Hct': [hct, [0,100], '%'], 
            'Hgb': [hgb, [0,30], 'g/dL'], 
            'Chloride': [cl, [0,200], 'mEq/L'],
            'Bicarbonate': [bic, [0,66], 'mEq/L'],
            'ALT': [alt, [0,11000], 'IU/L'],
            'ALP': [alp, [0,4000], 'IU/L'],
            'AST': [ast, [0,22000], 'IU/L'],
            'Albumin': [alb, [0,10], 'g/dL'],
            'Lactate': [lac, [0,33], 'mmol/L'],
            'LDH': [ld, [0,35000], 'IU/L'],
            'SG Urine': [usg, [0,2], ''],
            'pH Urine': [ph_ur, [0,14], ''],
            'pH Blood': [ph_bl, [0,14], ''],
            'PO2': [po2, [0,770], 'mmHg'],
            'PCO2': [pco2, [0,220], 'mmHg'],
            'Total CO2': [tco2, [0,65], 'mEq/L'],
            'Base Excess': [be, [-31, 28], 'mEq/L'],
            'Monocytes': [monos, [0,100], '%'],
            'Basophils': [baso, [0,100], '%'],
            'Eoisinophils': [eos, [0,100], '%'],
            'Neutrophils': [neuts, [0,100], '%'],
            'Lymphocytes': [lym_per, [0,100], '%'],
            'Lymphocytes (Absolute)': [lym_abs, [0,25000], '#/uL'],
            'PT': [pt, [0,150], 'sec'],
            'PTT': [ptt, [0,150], 'sec'],
            'INR': [inr, [0,150], ''],
            'Anion Gap': [agap, [0,55], 'mg/dL'],
            'BUN': [bun, [0,275], 'mEq/L'],
            'Creatinine Blood': [cr_bl, [0,66], 'mg/dL'],
            'Creatinine Urine': [cr_ur, [0,650], 'mg/dL'],
            'MCH': [mch, [0,50], 'pg'],
            'MCHC': [mchc, [0,50], '%'],
            'MCV': [mcv, [0,150], 'fL'],
            'RDW': [rdw, [0,37], '%'],
            'Platelet Count': [plt, [0,2200], 'K/uL'],
            'RBC': [rbc, [0,14], 'm/uL']
            }

for k, v in features.items():
    print (k)
    ev_k = pd.concat((ch.loc[ch.ITEMID.isin(v[0])], la.loc[la.ITEMID.isin(v[0])]))
    ev_k = ev_k.loc[(ev_k.VALUENUM>=v[1][0])&(ev_k.VALUENUM<=v[1][1])]
    ev_k['NAME'] = k
    ev_k['VALUEUOM'] = v[2]
    ev_k['VALUE'] = None
    assert (ev_k.VALUENUM.isna().sum()==0)
    events = pd.concat([events, ev_k])
del ev_k

# Free some memory.
del ch, la

# Extract outputevents.
oe = pd.read_csv(os.path.join(RAW_DATA_PATH,'OUTPUTEVENTS.csv'), 
                 usecols = ['ICUSTAY_ID', 'ITEMID', 'CHARTTIME', 'VALUE', 'VALUEUOM'])
oe = oe.loc[oe.VALUE.notna()]
oe['VALUENUM'] = oe.VALUE
oe.VALUE = None
oe = oe.loc[oe.ICUSTAY_ID.isin(icu.ICUSTAY_ID)]
oe.ICUSTAY_ID = oe.ICUSTAY_ID.astype(int)
oe['TABLE'] = 'output'

# Extract information about output items from D_ITEMS.csv.
items = pd.read_csv(os.path.join(RAW_DATA_PATH,'D_ITEMS.csv'), 
                    usecols=['ITEMID', 'LABEL', 'ABBREVIATION', 'UNITNAME', 'PARAM_TYPE'])
items.loc[items.LABEL.isna(), 'LABEL'] = ''
items.LABEL = items.LABEL.str.lower()
oeitems = oe[['ITEMID']].drop_duplicates()
oeitems = oeitems.merge(items, on='ITEMID', how='left')

# Extract multiple events. Replace outliers with median.
uf = [40286]
keys = ['urine', 'foley', 'void', 'nephrostomy', 'condom', 'drainage bag']
cond = pd.concat([oeitems.LABEL.str.contains(k) for k in keys], axis=1).any(axis='columns')
ur = list(oeitems.loc[cond].ITEMID)
keys = ['stool', 'fecal', 'colostomy', 'ileostomy', 'rectal']
cond = pd.concat([oeitems.LABEL.str.contains(k) for k in keys], axis=1).any(axis='columns')
st = list(oeitems.loc[cond].ITEMID)
ct = list(oeitems.loc[oeitems.LABEL.str.contains('chest tube')].ITEMID) + [226593, 226590, 226591, 226595, 226592]
gs = [40059, 40052, 226576, 226575, 226573, 40051, 226630]
ebl = [40064, 226626, 40491, 226629]
em = [40067, 226571, 40490, 41015, 40427]
jp = list(oeitems.loc[oeitems.LABEL.str.contains('jackson')].ITEMID)
res = [227510, 227511, 42837, 43892, 44909, 44959]
pre = [40060, 226633]

features = {'Ultrafiltrate': [uf, [0,7000],'mL'],
            'Urine': [ur, [0,2500], 'mL'],
            'Stool': [st, [0,4000], 'mL'],
            'Chest Tube': [ct, [0,2500], 'mL'],
            'Gastric': [gs, [0,4000], 'mL'],
            'EBL': [ebl, [0,10000], 'mL'],
#             'Pre-admission': [pre, [0,13000], 'mL'], # Repeated by mistake.
            'Emesis': [em, [0,2000], 'mL'],
            'Jackson-Pratt': [jp, [0,2000], 'ml'],
            'Residual': [res, [0, 1050], 'mL'],
            'Pre-admission Output': [pre, [0, 13000], 'ml']
            }

for k, v in features.items():
    print (k)
    ev_k = oe.loc[oe.ITEMID.isin(v[0])]
    ind = (ev_k.VALUENUM>=v[1][0])&(ev_k.VALUENUM<=v[1][1])
    med = ev_k.VALUENUM.loc[ind].median()
    ev_k.loc[~ind, 'VALUENUM'] = med
    ev_k['NAME'] = k
    ev_k['VALUEUOM'] = v[2]
    events = pd.concat([events, ev_k])
del ev_k

# Extract CV and MV inputevents.
ie_cv = pd.read_csv(os.path.join(RAW_DATA_PATH,'INPUTEVENTS_CV.csv'),
    usecols = ['ICUSTAY_ID', 'ITEMID', 'CHARTTIME', 
               'AMOUNT', 'AMOUNTUOM'])
ie_cv['TABLE'] = 'input_cv'
ie_cv = ie_cv.loc[ie_cv.AMOUNT.notna()]
ie_cv = ie_cv.loc[ie_cv.ICUSTAY_ID.isin(icu.ICUSTAY_ID)]
ie_cv.CHARTTIME = pd.to_datetime(ie_cv.CHARTTIME)

ie_mv = pd.read_csv(os.path.join(RAW_DATA_PATH,'INPUTEVENTS_MV.csv'),
    usecols = ['ICUSTAY_ID', 'ITEMID', 'STARTTIME', 'ENDTIME',
               'AMOUNT', 'AMOUNTUOM'])
ie_mv = ie_mv.loc[ie_mv.ICUSTAY_ID.isin(icu.ICUSTAY_ID)]

# Split MV intervals hourly.
ie_mv.STARTTIME = pd.to_datetime(ie_mv.STARTTIME)
ie_mv.ENDTIME = pd.to_datetime(ie_mv.ENDTIME)
ie_mv['TD'] = ie_mv.ENDTIME - ie_mv.STARTTIME
new_ie_mv = ie_mv.loc[ie_mv.TD<=pd.Timedelta(1,'h')].drop(columns=['STARTTIME', 'TD'])
ie_mv = ie_mv.loc[ie_mv.TD>pd.Timedelta(1,'h')]
new_rows = []
for _,row in tqdm(ie_mv.iterrows()):
    icuid, iid, amo, uom, stm, td = row.ICUSTAY_ID, row.ITEMID, row.AMOUNT, row.AMOUNTUOM, row.STARTTIME, row.TD
    td = td.total_seconds()/60
    num_hours = td // 60
    hour_amount = 60*amo/td
    for i in range(1,int(num_hours)+1):
        new_rows.append([icuid, iid, stm+pd.Timedelta(i,'h'), hour_amount, uom])
    rem_mins = td % 60
    if rem_mins>0:
        new_rows.append([icuid, iid, row['ENDTIME'], rem_mins*amo/td, uom])
new_rows = pd.DataFrame(new_rows, columns=['ICUSTAY_ID', 'ITEMID', 'ENDTIME', 'AMOUNT', 'AMOUNTUOM'])
new_ie_mv = pd.concat((new_ie_mv, new_rows))
ie_mv = new_ie_mv.copy()
del new_ie_mv
ie_mv['TABLE'] = 'input_mv' 
ie_mv.rename(columns={'ENDTIME':'CHARTTIME'}, inplace=True)

# Combine CV and MV inputevents.
ie = pd.concat((ie_cv, ie_mv))
del ie_cv, ie_mv
ie.rename(columns={'AMOUNT':'VALUENUM', 'AMOUNTUOM':'VALUEUOM'}, inplace=True)
events.CHARTTIME = pd.to_datetime(events.CHARTTIME)

# Convert mcg->mg, L->ml.
ind = (ie.VALUEUOM=='mcg')
ie.loc[ind, 'VALUENUM'] = ie.loc[ind, 'VALUENUM']*0.001
ie.loc[ind, 'VALUEUOM'] = 'mg'
ind = (ie.VALUEUOM=='L')
ie.loc[ind, 'VALUENUM'] = ie.loc[ind, 'VALUENUM']*1000
ie.loc[ind, 'VALUEUOM'] = 'ml'

# Extract Vasopressin events. Remove outliers.
vaso = [30051, 222315]
ev_vaso = ie.loc[ie.ITEMID.isin(vaso)]
ind1 = (ev_vaso.VALUENUM==0)
ind2 = ev_vaso.VALUEUOM.isin(['U','units'])
ind3 = (ev_vaso.VALUENUM>=0)&(ev_vaso.VALUENUM<=400)
ind = ((ind2&ind3)|ind1)
med = ev_vaso.VALUENUM.loc[ind].median()
ev_vaso.loc[~ind, 'VALUENUM'] = med
ev_vaso['VALUEUOM'] = 'units'
ev_vaso['NAME'] = 'Vasopressin'
events = pd.concat([events, ev_vaso])
del ev_vaso

# Extract Vancomycin events. Convert dose,g to mg. Remove outliers.
vanc = [225798]
ev_vanc = ie.loc[ie.ITEMID.isin(vanc)]
ind = ev_vanc.VALUEUOM.isin(['mg'])
ev_vanc.loc[ind, 'VALUENUM'] = ev_vanc.loc[ind, 'VALUENUM']*0.001 
ev_vanc['VALUEUOM'] = 'g'
ind = (ev_vanc.VALUENUM>=0)&(ev_vanc.VALUENUM<=8)
med = ev_vanc.VALUENUM.loc[ind].median()
ev_vanc.loc[~ind, 'VALUENUM'] = med
ev_vanc['NAME'] = 'Vacomycin'
events = pd.concat([events, ev_vanc])
del ev_vanc

# Extract Calcium Gluconate events. Convert units. Remove outliers.
cagl = [30023, 221456, 227525, 42504, 43070, 45699, 46591, 44346, 46291]
ev_cagl = ie.loc[ie.ITEMID.isin(cagl)]
ind = ev_cagl.VALUEUOM.isin(['mg'])
ev_cagl.loc[ind, 'VALUENUM'] = ev_cagl.loc[ind, 'VALUENUM']*0.001 
ind1 = (ev_cagl.VALUENUM==0)
ind2 = ev_cagl.VALUEUOM.isin(['mg', 'gm', 'grams'])
ind3 = (ev_cagl.VALUENUM>=0)&(ev_cagl.VALUENUM<=200)
ind = (ind2&ind3)|ind1
med = ev_cagl.VALUENUM.loc[ind].median()
ev_cagl.loc[~ind, 'VALUENUM'] = med
ev_cagl['VALUEUOM'] = 'g'
ev_cagl['NAME'] = 'Calcium Gluconate'
events = pd.concat([events, ev_cagl])
del ev_cagl

# Extract Furosemide events. Remove outliers.
furo = [30123, 221794, 228340]
ev_furo = ie.loc[ie.ITEMID.isin(furo)]
ind1 = (ev_furo.VALUENUM==0)
ind2 = (ev_furo.VALUEUOM=='mg')
ind3 = (ev_furo.VALUENUM>=0)&(ev_furo.VALUENUM<=250)
ind = ind1|(ind2&ind3)
med = ev_furo.VALUENUM.loc[ind].median()
ev_furo.loc[~ind, 'VALUENUM'] = med
ev_furo['VALUEUOM'] = 'mg'
ev_furo['NAME'] = 'Furosemide'
events = pd.concat([events, ev_furo])
del ev_furo

# Extract Famotidine events. Remove outliers.
famo = [225907]
ev_famo = ie.loc[ie.ITEMID.isin(famo)]
ind1 = (ev_famo.VALUENUM==0)
ind2 = (ev_famo.VALUEUOM=='dose')
ind3 = (ev_famo.VALUENUM>=0)&(ev_famo.VALUENUM<=1)
ind = ind1|(ind2&ind3)
med = ev_famo.VALUENUM.loc[ind].median()
ev_famo.loc[~ind, 'VALUENUM'] = med
ev_famo['VALUEUOM'] = 'dose'
ev_famo['NAME'] = 'Famotidine'
events = pd.concat([events, ev_famo])
del ev_famo

# Extract Piperacillin events. Convert units. Remove outliers.
pipe = [225893, 225892]
ev_pipe = ie.loc[ie.ITEMID.isin(pipe)]
ind1 = (ev_pipe.VALUENUM==0)
ind2 = (ev_pipe.VALUEUOM=='dose')
ind3 = (ev_pipe.VALUENUM>=0)&(ev_pipe.VALUENUM<=1)
ind = ind1|(ind2&ind3)
med = ev_pipe.VALUENUM.loc[ind].median()
ev_pipe.loc[~ind, 'VALUENUM'] = med
ev_pipe['VALUEUOM'] = 'dose'
ev_pipe['NAME'] = 'Piperacillin'
events = pd.concat([events, ev_pipe])
del ev_pipe

# Extract Cefazolin events. Convert units. Remove outliers.
cefa = [225850]
ev_cefa = ie.loc[ie.ITEMID.isin(cefa)]
ind1 = (ev_cefa.VALUENUM==0)
ind2 = (ev_cefa.VALUEUOM=='dose')
ind3 = (ev_cefa.VALUENUM>=0)&(ev_cefa.VALUENUM<=2)
ind = ind1|(ind2&ind3)
med = ev_cefa.VALUENUM.loc[ind].median()
ev_cefa.loc[~ind, 'VALUENUM'] = med 
ev_cefa['VALUEUOM'] = 'dose'
ev_cefa['NAME'] = 'Cefazolin'
events = pd.concat([events, ev_cefa])
del ev_cefa

# Extract Fiber events. Remove outliers.
fibe = [225936, 30166, 30073, 227695, 30088, 225928, 226051, 
        226050, 226048, 45381, 45597, 227699, 227696, 44218, 
        45406, 44675, 226049, 44202, 45370, 227698, 226027, 
        42106, 43994, 45865, 44318, 42091, 44699, 44010, 43134, 
        44045, 43088, 42641, 45691, 45515, 45777, 42663, 42027, 
        44425, 45657, 45775, 44631, 44106, 42116, 44061, 44887, 
        42090, 42831, 45541, 45497, 46789, 44765, 42050]
ev_fibe = ie.loc[ie.ITEMID.isin(fibe)]
ind1 = (ev_fibe.VALUENUM==0)
ind2 = (ev_fibe.VALUEUOM=='ml')
ind3 = (ev_fibe.VALUENUM>=0)&(ev_fibe.VALUENUM<=1600)
ind = ind1|(ind2&ind3)
med = ev_fibe.VALUENUM.loc[ind].median()
ev_fibe.loc[~ind, 'VALUENUM'] = med 
ev_fibe['NAME'] = 'Fiber'
ev_fibe['VALUEUOM'] = 'ml'
events = pd.concat([events, ev_fibe])
del ev_fibe

# Extract Pantoprazole events. Remove outliers.
pant = [225910, 40549, 41101, 41583, 44008, 40700, 40550]
ev_pant = ie.loc[ie.ITEMID.isin(pant)]
ind = (ev_pant.VALUENUM>0)
ev_pant.loc[ind, 'VALUENUM'] = 1
ind = (ev_pant.VALUENUM>=0)
med = ev_pant.VALUENUM.loc[ind].median()
ev_pant.loc[~ind, 'VALUENUM'] = med
ev_pant['NAME'] = 'Pantoprazole'
ev_pant['VALUEUOM'] = 'dose'
events = pd.concat([events, ev_pant])
del ev_pant

# Extract Magnesium Sulphate events. Remove outliers.
masu = [222011, 30027, 227524]
ev_masu = ie.loc[ie.ITEMID.isin(masu)]
ind = (ev_masu.VALUEUOM=='mg')
ev_masu.loc[ind, 'VALUENUM'] = ev_masu.loc[ind, 'VALUENUM']*0.001
ind1 = (ev_masu.VALUENUM==0)
ind2 = ev_masu.VALUEUOM.isin(['gm', 'grams', 'mg'])
ind3 = (ev_masu.VALUENUM>=0)&(ev_masu.VALUENUM<=125)
ind = ind1|(ind2&ind3)
med = ev_masu.VALUENUM.loc[ind].median()
ev_masu.loc[~ind, 'VALUENUM'] = med 
ev_masu['VALUEUOM'] = 'g'
ev_masu['NAME'] = 'Magnesium Sulphate'
events = pd.concat([events, ev_masu])
del ev_masu

# Extract Potassium Chloride events. Remove outliers.
poch = [30026, 225166, 227536]
ev_poch = ie.loc[ie.ITEMID.isin(poch)]
ind1 = (ev_poch.VALUENUM==0)
ind2 = ev_poch.VALUEUOM.isin(['mEq', 'mEq.'])
ind3 = (ev_poch.VALUENUM>=0)&(ev_poch.VALUENUM<=501)
ind = ind1|(ind2&ind3)
med = ev_poch.VALUENUM.loc[ind].median()
ev_poch.loc[~ind, 'VALUENUM'] = med 
ev_poch['VALUEUOM'] = 'mEq'
ev_poch['NAME'] = 'KCl'
events = pd.concat([events, ev_poch])
del ev_poch

# Extract multiple events. Remove outliers.
mida = [30124, 221668]
prop = [30131, 222168]
albu25 = [220862, 30009]
albu5 = [220864, 30008]
ffpl = [30005, 220970]
lora = [30141, 221385]
mosu = [30126, 225154]
game = [30144, 225799]
lari = [30021, 225828]
milr = [30125, 221986]
crys = [30101, 226364, 30108, 226375]
hepa = [30025, 225975, 225152]
prbc = [30001, 225168, 30104, 226368, 227070]
poin = [30056, 226452, 30109, 226377]
neos = [30128, 221749, 30127]
pigg = [226089, 30063]
nigl = [30121, 222056, 30049]
nipr = [30050, 222051]
meto = [225974]
nore = [30120, 221906, 30047]
coll = [30102, 226365, 30107, 226376]
hyzi = [221828]
gtfl = [226453, 30059]
hymo = [30163, 221833]
fent = [225942, 30118, 221744, 30149]
inre = [30045, 223258, 30100]
inhu = [223262]
ingl = [223260]
innp = [223259]
nana = [30140]
d5wa = [30013, 220949]
doth = [30015, 225823, 30060, 225825, 220950, 30016, 
        30061, 225827, 225941, 30160, 220952, 30159, 
        30014, 30017, 228142, 228140, 45360, 228141, 
        41550]
nosa = [225158, 30018]
hans = [30020, 225159]
stwa = [225944, 30065]
frwa = [30058, 225797, 41430, 40872, 41915, 43936, 
        41619, 42429, 44492, 46169, 42554]
solu = [225943]
dopa = [30043, 221662]
epin = [30119, 221289, 30044]
amio = [30112, 221347, 228339, 45402]
tpnu = [30032, 225916, 225917, 30096]
msbo = [227523]
pcbo = [227522]
prad = [30054, 226361]

features = {'Midazolam': [mida, [0, 500], 'mg'],
            'Propofol': [prop, [0, 12000], 'mg'],
            'Albumin 25%': [albu25, [0, 750], 'ml'],
            'Albumin 5%': [albu5, [0, 1300], 'ml'],
            'Fresh Frozen Plasma': [ffpl, [0, 33000], 'ml'],
            'Lorazepam': [lora, [0, 300], 'mg'],
            'Morphine Sulfate': [mosu, [0, 4000], 'mg'],
            'Gastric Meds': [game, [0, 7000], 'ml'],
            'Lactated Ringers': [lari, [0, 17000], 'ml'],
            'Milrinone': [milr, [0, 50], 'ml'],
            'OR/PACU Crystalloid': [crys, [0, 22000], 'ml'],
            'Packed RBC': [prbc, [0, 17250], 'ml'],
            'PO intake': [poin, [0, 11000], 'ml'],
            'Neosynephrine': [neos, [0, 1200], 'mg'],
            'Piggyback': [pigg, [0, 1000], 'ml'],
            'Nitroglycerine': [nigl, [0, 350], 'mg'],
            'Nitroprusside': [nipr, [0, 430], 'mg'],
            'Metoprolol': [meto, [0, 151], 'mg'],
            'Norepinephrine': [nore, [0, 80], 'mg'],
            'Colloid': [coll, [0, 20000], 'ml'],
            'Hydralazine': [hyzi, [0, 80], 'mg'],
            'GT Flush': [gtfl, [0, 2100], 'ml'],
            'Hydromorphone': [hymo, [0, 125], 'mg'],
            'Fentanyl': [fent, [0, 20], 'mg'],
            'Insulin Regular': [inre, [0, 1500], 'units'],
            'Insulin Humalog': [inhu, [0, 340], 'units'],
            'Insulin largine': [ingl, [0, 150], 'units'],
            'Insulin NPH': [innp, [0, 100], 'units'],
            'Unknown': [nana, [0, 1100], 'ml'],
            'D5W': [d5wa, [0,11000], 'ml'],
            'Dextrose Other': [doth, [0,4000], 'ml'],
            'Normal Saline': [nosa, [0, 11000], 'ml'],
            'Half Normal Saline': [hans, [0, 2000], 'ml'],
            'Sterile Water': [stwa, [0, 10000], 'ml'],
            'Free Water': [frwa, [0, 2500], 'ml'],
            'Solution': [solu, [0, 1500], 'ml'],
            'Dopamine': [dopa, [0, 1300], 'mg'],
            'Epinephrine': [epin, [0, 100], 'mg'],
            'Amiodarone': [amio, [0, 1200], 'mg'],
            'TPN': [tpnu, [0, 1600], 'ml'],
            'Magnesium Sulfate (Bolus)': [msbo, [0, 250], 'ml'],
            'KCl (Bolus)': [pcbo, [0, 500], 'ml'],
            'Pre-admission Intake': [prad, [0, 30000], 'ml']
            }

for k, v in features.items():
    print (k)
    ev_k = ie.loc[ie.ITEMID.isin(v[0])]
    ind = (ev_k.VALUENUM>=v[1][0])&(ev_k.VALUENUM<=v[1][1])
    med = ev_k.VALUENUM.loc[ind].median()
    ev_k.loc[~ind, 'VALUENUM'] = med
    ev_k['NAME'] = k
    ev_k['VALUEUOM'] = v[2]
    events = pd.concat([events, ev_k])
del ev_k

# Extract heparin events. (Missed earlier.)
ev_k = ie.loc[ie.ITEMID.isin(hepa)]
ind1 = ev_k.VALUEUOM.isin(['U', 'units'])
ind2 = (ev_k.VALUENUM>=0)&(ev_k.VALUENUM<=25300)
ind = (ind1&ind2)
med = ev_k.VALUENUM.loc[ind].median()
ev_k.loc[~ind, 'VALUENUM'] = med
ev_k['NAME'] = 'Heparin'
ev_k['VALUEUOM'] = 'units'
events = pd.concat([events, ev_k])
del ev_k

# Extract weight events from MV inputevents.
ie_mv = pd.read_csv(os.path.join(RAW_DATA_PATH,'INPUTEVENTS_MV.csv'), 
                    usecols = ['ICUSTAY_ID', 'STARTTIME', 'PATIENTWEIGHT'])
ie_mv = ie_mv.drop_duplicates()
ie_mv = ie_mv.loc[ie_mv.ICUSTAY_ID.isin(icu.ICUSTAY_ID)]
ie_mv.rename(columns={'STARTTIME':'CHARTTIME', 'PATIENTWEIGHT':'VALUENUM'}, inplace=True)
ie_mv = ie_mv.loc[(ie_mv.VALUENUM>=0)&(ie_mv.VALUENUM<=300)]
ie_mv['VALUEUOM'] = 'kg'
ie_mv['NAME'] = 'Weight'
events = pd.concat([events, ie_mv])[['HADM_ID', 'ICUSTAY_ID', 'CHARTTIME', 'VALUENUM', 'TABLE', 'NAME']]
del ie_mv

# Convert times to type datetime.
events.CHARTTIME = pd.to_datetime(events.CHARTTIME)
icu.INTIME = pd.to_datetime(icu.INTIME)
icu.OUTTIME = pd.to_datetime(icu.OUTTIME)

# Assign ICUSTAY_ID to rows without it. Remove rows that can't be assigned one.
icu['icustay_times'] = icu.apply(lambda x:[x.ICUSTAY_ID, x.INTIME, x.OUTTIME], axis=1)
adm_icu_times = icu.groupby('HADM_ID').agg({'icustay_times':list}).reset_index()
icu.drop(columns=['icustay_times'], inplace=True)
events = events.merge(adm_icu_times, on=['HADM_ID'], how='left')
idx = events.ICUSTAY_ID.isna()
tqdm.pandas()
def f(x):
    chart_time = x.CHARTTIME
    for icu_times in x.icustay_times:
        if icu_times[1]<=chart_time<=icu_times[2]:
            return icu_times[0]
events.loc[idx, 'ICUSTAY_ID'] = (events.loc[idx]).progress_apply(f, axis=1)
events.drop(columns=['icustay_times'], inplace=True)
events = events.loc[events.ICUSTAY_ID.notna()]
events.drop(columns=['HADM_ID'], inplace=True)

# Filter icu table.
icu = icu.loc[icu.ICUSTAY_ID.isin(events.ICUSTAY_ID)]

# Get rel_charttime in minutes.
events = events.merge(icu[['ICUSTAY_ID', 'INTIME']], on='ICUSTAY_ID', how='left')
events['rel_charttime'] = events.CHARTTIME-events.INTIME
events.drop(columns=['INTIME', 'CHARTTIME'], inplace=True)
events.rel_charttime = events.rel_charttime.dt.total_seconds()//60

# Save current icu table.
icu_full = icu.copy()

# Get icustays which lasted for atleast 24 hours.
icu = icu.loc[(icu.OUTTIME-icu.INTIME)>=pd.Timedelta(24,'h')]

# Get icustays with patient alive for atleast 24 hours.
adm = pd.read_csv(os.path.join(RAW_DATA_PATH,'ADMISSIONS.csv'),
                  usecols=['HADM_ID', 'DEATHTIME'])
icu = icu.merge(adm, on='HADM_ID', how='left')
icu.DEATHTIME = pd.to_datetime(icu.DEATHTIME)
icu = icu.loc[((icu.DEATHTIME-icu.INTIME)>=pd.Timedelta(24,'h'))|icu.DEATHTIME.isna()]

# Get icustays with aleast one event in first 24h.
icu = icu.loc[icu.ICUSTAY_ID.isin(events.loc[events.rel_charttime<24*60].ICUSTAY_ID)]

# Get sup and unsup icustays.
all_icustays = np.array(icu_full.ICUSTAY_ID)
sup_icustays = np.array(icu.ICUSTAY_ID)
unsup_icustays = np.setdiff1d(all_icustays, sup_icustays)
all_icustays = np.concatenate((sup_icustays, unsup_icustays), axis=-1)

# Rename some columns.
events.rename(columns={'rel_charttime':'minute', 'NAME':'variable', 
                       'VALUENUM':'value', 'ICUSTAY_ID':'ts_id'}, inplace=True)

# Add gender and age.
icu_full.rename(columns={'ICUSTAY_ID':'ts_id'}, inplace=True)
data_age = icu_full[['ts_id', 'AGE']]
data_age['variable'] = 'Age'
data_age.rename(columns={'AGE':'value'}, inplace=True)
data_gen = icu_full[['ts_id', 'GENDER']]
data_gen.loc[data_gen.GENDER=='M', 'GENDER'] = 0
data_gen.loc[data_gen.GENDER=='F', 'GENDER'] = 1
data_gen['variable'] = 'Gender'
data_gen.rename(columns={'GENDER':'value'}, inplace=True)
data = pd.concat((data_age, data_gen), ignore_index=True)
data['minute'] = 0
events = pd.concat((data, events), ignore_index=True)

# Drop duplicate events.
events.drop_duplicates(inplace=True)

# Add mortality label.
adm = pd.read_csv(os.path.join(RAW_DATA_PATH,'ADMISSIONS.csv'), 
                  usecols=['HADM_ID', 'HOSPITAL_EXPIRE_FLAG'])
oc = icu_full[['ts_id', 'HADM_ID', 'SUBJECT_ID']].merge(adm, on='HADM_ID', how='left')
oc = oc.rename(columns={'HOSPITAL_EXPIRE_FLAG': 'in_hospital_mortality'})


# Get train-valid-test split for sup task.
all_sup_subjects = icu.SUBJECT_ID.unique()
np.random.seed(0)
np.random.shuffle(all_sup_subjects)
S = len(all_sup_subjects)
bp1, bp2 = int(0.64*S), int(0.8*S)
train_sub = all_sup_subjects[:bp1]
valid_sub = all_sup_subjects[bp1:bp2]
test_sub = all_sup_subjects[bp2:]
icu.rename(columns={'ICUSTAY_ID':'ts_id'}, inplace=True)
train_ids = np.array(icu.loc[icu.SUBJECT_ID.isin(train_sub)].ts_id)
valid_ids = np.array(icu.loc[icu.SUBJECT_ID.isin(valid_sub)].ts_id)
test_ids = np.array(icu.loc[icu.SUBJECT_ID.isin(test_sub)].ts_id)

# Filter columns.
events = events[['ts_id', 'minute', 'variable', 'value', 'TABLE']]

# Aggregate data.
events['value'] = events['value'].astype(float)
events.loc[events['TABLE'].isna(), 'TABLE'] = 'N/A'
events = events.groupby(['ts_id', 'minute', 'variable']).agg(
                {'value':'mean', 'TABLE':'unique'}).reset_index()
def f(x):
    if len(x)==0:
        return x[0]
    else:
        return ','.join(x)
events['TABLE'] = events['TABLE'].apply(f)

# Save data.
os.makedirs('../data/processed', exist_ok=True)
pickle.dump([events, oc, train_ids, valid_ids, test_ids], 
            open('../data/processed/mimic_iii.pkl','wb'))
