import pandas as pd
import numpy as np
import requests
import gspread
from datetime import datetime
from oauth2client.service_account import ServiceAccountCredentials
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================================
# 0. 參數與環境設定
# ==========================================
TARGET_SHEET_URL = 'https://docs.google.com/spreadsheets/d/1seGpSiQSUCZMgEqs66nsycI5GLvqTiam8mLDry5G4t8/edit?usp=sharing'
SERVICE_ACCOUNT_FILE = 'service_account.json' 

# ==========================================
# 1. 資料抓取模組
# ==========================================
def fetch_all_data():
    print("🚀 正在聯網抓取最新資料...")
    # 幼兒園
    df_k = pd.read_csv("https://stats.moe.gov.tw/files/opendata/edu_B_1_4.csv", encoding='utf-8-sig')
    df_k = df_k[df_k['縣市別'] == '臺中市'][['學年度', '幼兒園[人]']]
    df_k['Year'] = df_k['學年度'] + 1911
    df_k = df_k.rename(columns={'幼兒園[人]': 'Kindergarten_Enrollment'})
    
    # 疾管署 - 健保
    df_nhi = pd.read_csv("https://od.cdc.gov.tw/eic/NHI_EnteroviralInfection.csv", encoding='utf-8-sig')
    df_nhi = df_nhi[(df_nhi['縣市'] == '台中市') & (df_nhi['年齡別'].isin(['0~2', '3~6']))]
    df_nhi = df_nhi.groupby(['年', '週'])[['腸病毒健保就診人次']].sum().reset_index()
    
    # 疾管署 - 急診 (Target)
    df_er = pd.read_csv("https://od.cdc.gov.tw/eic/RODS_EnteroviralInfection.csv", encoding='utf-8-sig')
    df_er = df_er[(df_er['縣市'] == '台中市') & (df_er['年齡別'].isin(['0', '1~3', '4~6']))]
    df_er = df_er.groupby(['年', '週'])[['腸病毒急診就診人次']].sum().reset_index()

    return df_er, df_nhi, df_k

# ==========================================
# 2. 資料處理
# ==========================================
def process_data(df_er, df_nhi, df_k):
    df_er = df_er.rename(columns={'年': 'Year', '週': 'Week', '腸病毒急診就診人次': 'EV_ER_Cases'})
    df_nhi = df_nhi.rename(columns={'年': 'Year', '週': 'Week', '腸病毒健保就診人次': 'EV_NHI_Cases'})
    
    df = pd.merge(df_er, df_nhi, on=['Year', 'Week'], how='left')
    df = pd.merge(df, df_k[['Year', 'Kindergarten_Enrollment']], on='Year', how='left')
    df = df.sort_values(['Year', 'Week']).reset_index(drop=True)
    
    # Lag 3 邏輯 (預測下週時，使用兩週前資料)
    df['Lag3_ER'] = df['EV_ER_Cases'].shift(3)
    df['Lag4_ER'] = df['EV_ER_Cases'].shift(4)
    df['Lag3_NHI'] = df['EV_NHI_Cases'].shift(3)
    df['Kindergarten_Enrollment'] = df['Kindergarten_Enrollment'].ffill()
    
    df['Week_Sin'] = np.sin(2 * np.pi * df['Week'] / 53)
    df['Week_Cos'] = np.cos(2 * np.pi * df['Week'] / 53)
    return df

# ==========================================
# 3. 模型預測核心
# ==========================================
def run_model_pipeline(df):
    features = ['Year', 'Week', 'Lag3_ER', 'Lag4_ER', 'Lag3_NHI', 'Kindergarten_Enrollment', 'Week_Sin', 'Week_Cos']
    target = 'EV_ER_Cases'
    
    train_df = df.dropna(subset=features + [target])
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(train_df[features], train_df[target])
    
    mae = round(mean_absolute_error(train_df[target], model.predict(train_df[features])), 2)
    
    # 預測下週
    now = datetime.now()
    _, cur_w, _ = now.isocalendar()
    t_year, t_week = (now.year, cur_w + 1) if cur_w < 53 else (now.year + 1, 1)
    
    latest = df.iloc[-1]
    input_v = pd.DataFrame([{
        'Year': t_year, 'Week': t_week,
        'Lag3_ER': latest['EV_ER_Cases'], 'Lag4_ER': df.iloc[-2]['EV_ER_Cases'],
        'Lag3_NHI': latest['EV_NHI_Cases'], 'Kindergarten_Enrollment': latest['Kindergarten_Enrollment'],
        'Week_Sin': np.sin(2 * np.pi * t_week / 53), 'Week_Cos': np.cos(2 * np.pi * t_week / 53)
    }])
    
    prediction = model.predict(input_v)[0]
    
    # 建立輸出結果
    pred_res = pd.DataFrame([{
        'Forecast_Timestamp': now.strftime('%Y-%m-%d %H:%M'),
        'Target_Period': f"{int(t_year)}W{int(t_week):02d}",
        'Predicted_ER_Cases': round(prediction, 2),
        'Model_MAE': mae,
        'Input_Ref_Week': f"{int(latest['Year'])}W{int(latest['Week']):02d}",
        'Ref_Actual_ER': latest['EV_ER_Cases'],
        'Ref_Actual_NHI': latest['EV_NHI_Cases']
    }])
    
    importances = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    return pred_res, importances

# ==========================================
# 4. Google Sheets 上傳模組 (修正欄位名稱遺失問題)
# ==========================================
def upload_to_sheets(pred_df, importance_df):
    print("📤 正在同步資料至 Google Sheets...")
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_name(SERVICE_ACCOUNT_FILE, scope)
    client = gspread.authorize(creds)
    sheet = client.open_by_url(TARGET_SHEET_URL)
    
    # --- 1. 處理「預測結果」分頁 ---
    try:
        ws_pred = sheet.worksheet("預測結果")
    except:
        ws_pred = sheet.add_worksheet(title="預測結果", rows="100", cols="10")
    
    # 檢查是否有標題列，若無或不對則寫入
    headers = pred_df.columns.tolist()
    current_values = ws_pred.get_all_values()
    
    if not current_values or current_values[0] != headers:
        # 如果是空的或標題不對，在第一列插入標題
        ws_pred.insert_row(headers, 1)
        print("💡 已自動建立/校正標題列。")
    
    # 附加數據
    ws_pred.append_rows(pred_df.values.tolist())

    # --- 2. 處理「模型監控」分頁 ---
    try:
        ws_stats = sheet.worksheet("模型監控")
    except:
        ws_stats = sheet.add_worksheet(title="模型監控", rows="100", cols="10")
    
    ws_stats.clear()
    # 寫入特徵重要性與標題
    ws_stats.update('A1', [['模型特徵重要性分析 (Feature Importance)']])
    ws_stats.update('A2', [importance_df.columns.tolist()]) # 欄位標題
    ws_stats.update('A3', importance_df.values.tolist()) # 數值內容
    
    print("✅ Sheets 更新成功！")

# ==========================================
# 5. 主程式執行
# ==========================================
if __name__ == "__main__":
    try:
        er, nhi, k = fetch_all_data()
        df_final = process_data(er, nhi, k)
        p_res, f_imp = run_model_pipeline(df_final)
        upload_to_sheets(p_res, f_imp)
        print(f"\n🎉 執行成功！下週預測值為: {p_res['Predicted_ER_Cases'].iloc[0]}")
    except Exception as e:
        print(f"❌ 錯誤: {e}")