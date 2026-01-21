import requests
import pandas as pd
import numpy as np
import json
import os
import gspread
from datetime import datetime, timedelta
from oauth2client.service_account import ServiceAccountCredentials
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# ==========================================
# 0. 參數與環境設定
# ==========================================
CWA_API_URL = "https://opendata.cwa.gov.tw/api/v1/rest/datastore/C-B0024-001"
CWA_TOKEN = os.getenv("CWA_TOKEN")
PM25_API_URL = "https://data.moenv.gov.tw/api/v2/aqx_p_322?api_key=4c89a32a-a214-461b-bf29-30ff32a61a8a&sort=monitordate%20desc&format=CSV"
TARGET_SHEET_URL = "https://docs.google.com/spreadsheets/d/1seGpSiQSUCZMgEqs66nsycI5GLvqTiam8mLDry5G4t8/edit?usp=sharing"
# LINE 機器人設定
LINE_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_USER = os.getenv("LINE_USER_ID")
GITHUB_REPO_URL = "https://raw.githubusercontent.com/za29517585/enterovirus_prediction/main"

HIST_DIR = "./hist_data/"

# ==========================================
# 1. 輔助函式：讀取或抓取資料
# ==========================================

def get_historical_or_fetch_new(file_name, fetch_func):
    """嘗試讀取歷史檔，並執行爬蟲抓取最新資料"""
    file_path = os.path.join(HIST_DIR, file_name)
    
    # 執行爬蟲獲取最新週資料 (這部分沿用您之前的 ETL 邏輯)
    fetch_func() 
    
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        print(f"⚠️ 找不到歷史檔案: {file_name}")
        return pd.DataFrame()

def fetch_all_source_data():
    print("🚀 正在同步所有來源資料 (CDC, MOE, CWA, MoENV)...")
    
    # A. 幼兒園人數
    df_k = pd.read_csv("https://stats.moe.gov.tw/files/opendata/edu_B_1_4.csv", encoding='utf-8-sig')
    df_k = df_k[df_k['縣市別'] == '臺中市'][['學年度', '幼兒園[人]']]
    df_k['Year'] = df_k['學年度'] + 1911
    df_k = df_k.rename(columns={'幼兒園[人]': 'Kindergarten_Enrollment'})

    # B. CDC 腸病毒資料
    df_nhi = pd.read_csv("https://od.cdc.gov.tw/eic/NHI_EnteroviralInfection.csv", encoding='utf-8-sig')
    df_nhi = df_nhi[(df_nhi['縣市'] == '台中市') & (df_nhi['年齡別'].isin(['0~2', '3~6']))]
    df_nhi = df_nhi.groupby(['年', '週'])[['腸病毒健保就診人次']].sum().reset_index()
    
    df_er = pd.read_csv("https://od.cdc.gov.tw/eic/RODS_EnteroviralInfection.csv", encoding='utf-8-sig')
    df_er = df_er[(df_er['縣市'] == '台中市') & (df_er['年齡別'].isin(['0', '1~3', '4~6']))]
    df_er = df_er.groupby(['年', '週'])[['腸病毒急診就診人次']].sum().reset_index()

    # C. 讀取氣象與 PM2.5 歷史存檔 (假設您之前的 ETL 已經跑過並存檔)
    df_temp = pd.read_csv(os.path.join(HIST_DIR, 'temp_hist.csv')) if os.path.exists(os.path.join(HIST_DIR, 'temp_hist.csv')) else pd.DataFrame()
    df_rh = pd.read_csv(os.path.join(HIST_DIR, 'rh_hist.csv')) if os.path.exists(os.path.join(HIST_DIR, 'rh_hist.csv')) else pd.DataFrame()
    df_pm = pd.read_csv(os.path.join(HIST_DIR, 'pm25_hist.csv')) if os.path.exists(os.path.join(HIST_DIR, 'pm25_hist.csv')) else pd.DataFrame()

    return df_er, df_nhi, df_k, df_temp, df_rh, df_pm

# ==========================================
# 2. 資料處理
# ==========================================
def process_data(df_er, df_nhi, df_k, df_temp, df_rh, df_pm):
    print("📊 整合特徵與歷史資料...")
    
    # 標準化欄位名稱以利合併
    df_er = df_er.rename(columns={'年': 'Year', '週': 'Week', '腸病毒急診就診人次': 'EV_ER'})
    df_nhi = df_nhi.rename(columns={'年': 'Year', '週': 'Week', '腸病毒健保就診人次': 'EV_NHI'})
    
    # 合併 CDC 資料
    df = pd.merge(df_er, df_nhi, on=['Year', 'Week'], how='outer')
    df['EV_Total_Cases'] = df['EV_ER'].fillna(0) + df['EV_NHI'].fillna(0)
    
    # 合併歷史氣象與 PM2.5 (這步最關鍵，決定了模型有沒有訓練樣本)
    if not df_temp.empty:
        df_temp = df_temp.rename(columns={'年': 'Year', '週次': 'Week', '臺中市氣溫_週平均': 'temp'})
        df = pd.merge(df, df_temp, on=['Year', 'Week'], how='left')
    
    if not df_rh.empty:
        df_rh = df_rh.rename(columns={'年': 'Year', '週次': 'Week', '臺中市相對溼度_週平均': 'rh'})
        df = pd.merge(df, df_rh, on=['Year', 'Week'], how='left')
        
    if not df_pm.empty:
        df_pm = df_pm.rename(columns={'年': 'Year', '週次': 'Week', '臺中市PM2.5_週平均': 'PM25'})
        df = pd.merge(df, df_pm, on=['Year', 'Week'], how='left')

    df = pd.merge(df, df_k[['Year', 'Kindergarten_Enrollment']], on='Year', how='left')
    
    # 排序並處理特徵
    df = df.sort_values(['Year', 'Week']).reset_index(drop=True)
    df['Lag3_Total'] = df['EV_Total_Cases'].shift(3)
    df['Lag4_Total'] = df['EV_Total_Cases'].shift(4)
    
    # 填充氣象缺值 (針對最新還沒湊滿一週的部分)
    for col in ['temp', 'rh', 'PM25', 'Kindergarten_Enrollment']:
        df[col] = df[col].ffill()

    # 週期特徵
    df['Week_Sin'] = np.sin(2 * np.pi * df['Week'] / 53)
    df['Week_Cos'] = np.cos(2 * np.pi * df['Week'] / 53)
    
    return df

# ==========================================
# 3. 模型訓練與預測 (加入檢查機制)
# ==========================================
def run_model_pipeline(df):
    features = ['Year', 'Week', 'Lag3_Total', 'Lag4_Total', 'temp', 'rh', 'PM25', 'Kindergarten_Enrollment', 'Week_Sin', 'Week_Cos']
    target = 'EV_Total_Cases'
    
    # 檢查訓練集是否為空
    train_df = df.dropna(subset=features + [target])
    
    if train_df.empty:
        raise ValueError("❌ 訓練資料集為空！請檢查歷史 CSV 檔 (temp_hist.csv 等) 是否正確存在於 ./hist/ 資料夾中。")

    print(f"📈 訓練樣本數: {len(train_df)}")
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(train_df[features], train_df[target])
    
    mae = round(mean_absolute_error(train_df[target], model.predict(train_df[features])), 2)
    
    # 預測下週
    now_taipei = datetime.now() + timedelta(hours=8)
    _, cur_w, _ = now_taipei.isocalendar()
    target_year, target_week = (now_taipei.year, cur_w + 1) if cur_w < 53 else (now_taipei.year + 1, 1)
    
    latest = df.iloc[-1]
    input_v = pd.DataFrame([{
        'Year': target_year, 'Week': target_week,
        'Lag3_Total': latest['EV_Total_Cases'], 
        'Lag4_Total': df.iloc[-2]['EV_Total_Cases'],
        'temp': latest['temp'], 'rh': latest['rh'], 'PM25': latest['PM25'],
        'Kindergarten_Enrollment': latest['Kindergarten_Enrollment'],
        'Week_Sin': np.sin(2 * np.pi * target_week / 53), 
        'Week_Cos': np.cos(2 * np.pi * target_week / 53)
    }])
    
    prediction = model.predict(input_v)[0]
    
    pred_res = pd.DataFrame([{
        'Forecast_Timestamp': now_taipei.strftime('%Y-%m-%d %H:%M'),
        'Target_Period': f"{int(target_year)}W{int(target_week):02d}",
        'Predicted_Total_Cases': round(prediction, 2),
        'Model_MAE': mae,
        'Input_Ref_Week': f"{int(latest['Year'])}W{int(latest['Week']):02d}",
        'Ref_Actual_Total': latest['EV_Total_Cases']
    }])
    
    importances = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_}).sort_values(by='Importance', ascending=False)
    
    return pred_res, importances

# ==========================================
# 新增功能：LINE 機器人推送通知
# ==========================================
def send_line_notification(prediction_val):
    if not LINE_TOKEN or not LINE_USER:
        print("⚠️ 找不到 LINE Token 或 User ID，跳過通知發送。")
        return

    url = "https://api.line.me/v2/bot/message/push"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LINE_TOKEN}"
    }

    # 判斷風險等級與訊息
    if prediction_val < 1040:
        status = "🟢 低風險（安全期）"
        msg = f"{status}\n下週預估人數：{prediction_val}\n下週為低風險期，建議維持一般洗手衛生習慣，落實正確洗手5步驟：「濕、搓（至少20秒）、沖、捧、擦」，以及防治腸病毒5口訣「勤洗手、足睡眠、多運動、洗玩具、及早治療」。"
        img_list = ["low1.jpg", "low2.jpg"]
    elif 1040 <= prediction_val <= 1300:
        status = "🟡 中風險（警覺期）"
        msg = f"{status}\n下週預估人數：{prediction_val}\n下週為中風險期，請各位家長要多注意自己就學的孩子們體溫及口腔有無出現小水泡，該提高警覺囉～"
        img_list = ["mid.jpg"]
    else:
        status = "🔴 高風險（流行期)"
        msg = f"{status}\n下週預估人數：{prediction_val}\n下週為高風險期，請記得做好個人防護及加強環境消毒（如 500ppm 漂白水），若小朋友有生病現象，記得要落實「生病不上學」，保護自己保護別人～"
        img_list = ["high.jpg"]

    # 封裝訊息內容
    messages = [{"type": "text", "text": msg}]
    
    # 加入圖片訊息 (LINE API 限制單次 Push 最多 5 則訊息)
    for img_name in img_list:
        img_url = f"{GITHUB_REPO_URL}/{img_name}"
        messages.append({
            "type": "image",
            "originalContentUrl": img_url,
            "previewImageUrl": img_url
        })

    payload = {
        "to": LINE_USER,
        "messages": messages
    }

    res = requests.post(url, headers=headers, data=json.dumps(payload))
    if res.status_code == 200:
        print(f"✅ LINE 通知已發送：{status}")
    else:
        print(f"❌ LINE 通知發送失敗: {res.text}")

# ==========================================
# 4. Google Sheets 上傳
# ==========================================
def upload_to_sheets(pred_df, importance_df):
    print("📤 正在透過環境變數同步資料至 Google Sheets...")
    
    # 1. 從環境變數獲取 JSON 字串
    gcp_sa_key_str = os.getenv("GCP_SA_KEY")
    
    if not gcp_sa_key_str:
        raise ValueError("❌ 找不到環境變數 GCP_SA_KEY，請檢查設定。")
    
    # 2. 將 JSON 字串解析為 Dictionary
    info = json.loads(gcp_sa_key_str)
    
    # 3. 使用 from_json_keyfile_dict 進行驗證
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_dict(info, scope)
    client = gspread.authorize(creds)
    
    sheet = client.open_by_url(TARGET_SHEET_URL)
    
    # --- 處理「預測結果」 ---
    try:
        ws_pred = sheet.worksheet("預測結果")
    except:
        ws_pred = sheet.add_worksheet(title="預測結果", rows="100", cols="10")
    
    headers = pred_df.columns.tolist()
    current_values = ws_pred.get_all_values()
    if not current_values or current_values[0] != headers:
        ws_pred.insert_row(headers, 1)
    
    ws_pred.append_rows(pred_df.values.tolist())

    # --- 處理「模型監控」 ---
    try:
        ws_stats = sheet.worksheet("模型監控")
    except:
        ws_stats = sheet.add_worksheet(title="模型監控", rows="100", cols="10")
    
    ws_stats.clear()
    ws_stats.update('A1', [['腸病毒預測模型 - 特徵重要性分析']])
    ws_stats.update('A2', [importance_df.columns.tolist()])
    ws_stats.update('A3', importance_df.values.tolist())
    print("✅ Sheets 更新完成！")

# ==========================================
# 5. 主程式
# ==========================================
if __name__ == "__main__":
    try:
        # 1. 抓取所有資料
        df_er, df_nhi, df_k, df_temp, df_rh, df_pm = fetch_all_source_data()
        # 2. 整合資料
        df_final = process_data(df_er, df_nhi, df_k, df_temp, df_rh, df_pm)
        # 3. 執行模型與預測
        p_res, f_imp = run_model_pipeline(df_final)
        # 4. 上傳 Google Sheets
        upload_to_sheets(p_res, f_imp)
        
        # --- 執行新功能：發送 LINE 通知 ---
        prediction_val = p_res['Predicted_Total_Cases'].iloc[0]
        send_line_notification(prediction_val)
        
        print(f"\n🎉 任務執行成功！預測人數為 {prediction_val}")
        
    except Exception as e:
        print(f"❌ 發生錯誤: {e}")