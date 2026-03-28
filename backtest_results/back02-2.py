import pandas as pd
import numpy as np
import pandas_ta as ta
from binance.client import Client
import warnings
import random
from multiprocessing import Pool, cpu_count
import os
import time
import pprint

warnings.filterwarnings('ignore')

# [1. 설정 영역]
client = Client("", "", {"verify": True, "timeout": 20}) 
SYMBOL = 'FARTCOINUSDT'
TOTAL_DAYS = 480      
BUFFER_DAYS = 60      
LEVERAGE = 5          

TAKER_FEE = 0.0005
SLIPPAGE = 0.001 

# GA 하이퍼 파라미터
POP_SIZE = 10000        
GENERATIONS = 500     
MUTATION_RATE = 0.12  
ELITE_SIZE = 250      
PATIENCE = 40         
MDD_LIMIT = 0.6       

# 최적화된 유전자 범위
GENE_BOUNDS = {
    'r_adx_limit': (15.0, 35.0), 
    't_adx_limit_normal': (20.0, 50.0), 
    't_adx_limit_strong': (35.0, 65.0), 
    'r_slope_max': (-5.0, 0.0),
    't_slope_min': (1.0, 15.0),    
    't_slope_strong': (3.0, 20.0),   
    't_tp_short_mult': (1.2, 5.0),   
    't_tp_mult': (4.0, 15.0),           
    'r_tp_mult': (1.5, 4.5),            
    'r_sl_mult': (0.0001, 0.01),        
    't_sl_base_normal': (0.0001, 0.006), 
    't_sl_base_strong': (0.0002, 0.012), 
    't_ts_mult': (0.0001, 0.005),   
    't_sl_activate': (0.01, 0.05),  
    'r_vol_limit': (0.1, 2.0),    
    't_vol_limit_normal': (0.1, 2.0),
    't_vol_limit_strong': (0.5, 3.5),
    'rsi_low': (30.0, 55.0),      
    'rsi_high': (45.0, 70.0),     
    't_rsi_max_normal': (60.0, 85.0), 
    't_rsi_min_normal': (15.0, 40.0),
    't_rsi_max_strong': (70.0, 95.0), 
    't_rsi_min_strong': (10.0, 35.0)  
}

INTERVALS = ['1h', '2h', '4h']
TF_KEYS = ['r_inter', 't_inter_normal', 't_inter_strong', 'atr_inter']

# [2. 데이터 준비]
def get_data(symbol, interval, days):
    for i in range(3):
        try:
            klines = client.futures_historical_klines(symbol, interval, f"{days} days ago UTC")
            df = pd.DataFrame(klines, columns=['ts', 'open', 'high', 'low', 'close', 'vol', 'ct', 'qv', 'tr', 'tb', 'tq', 'ig'])
            df['ts'] = pd.to_datetime(df['ts'], unit='ms')
            df[['open', 'high', 'low', 'close', 'vol']] = df[['open', 'high', 'low', 'close', 'vol']].astype(float)
            return df[['ts', 'open', 'high', 'low', 'close', 'vol']]
        except: time.sleep(2)
    return pd.DataFrame()

def prepare_full_data():
    print(f"🔄 {SYMBOL} 데이터 로딩 및 지표 계산...")
    df_raw = get_data(SYMBOL, '3m', TOTAL_DAYS + BUFFER_DAYS)
    if df_raw.empty: return None
    df_raw['vol_mean'] = df_raw['vol'].rolling(20).mean()

    for tf in INTERVALS:
        multiplier = 20 if tf == '1h' else 40 if tf == '2h' else 80
        df_raw[f'ma20_{tf}'] = ta.sma(df_raw['close'], length=20 * multiplier)
        
        df_tf = get_data(SYMBOL, tf, TOTAL_DAYS + BUFFER_DAYS)
        if not df_tf.empty:
            adx_series = ta.adx(df_tf['high'], df_tf['low'], df_tf['close'])['ADX_14']
            df_tf[f'adx_{tf}'] = adx_series
            df_tf[f'adx_slope_{tf}'] = adx_series.pct_change() * 100
            df_tf[f'atr_{tf}'] = ta.atr(df_tf['high'], df_tf['low'], df_tf['close'], length=14)
            df_tf[f'rsi_{tf}'] = ta.rsi(df_tf['close'], length=14)
            df_tf[f'vol_{tf}_mean'] = df_tf['vol'].rolling(20).mean()
            
            # BBW 계산 (iloc로 컬럼명 이슈 방지)
            bb = ta.bbands(df_tf['close'], length=20, std=2.0)
            bbl, bbm, bbu = bb.iloc[:, 0], bb.iloc[:, 1], bb.iloc[:, 2]
            df_tf[f'bbw_{tf}'] = (bbu - bbl) / (bbm + 1e-9)
            df_tf[f'bbw_slope_{tf}'] = df_tf[f'bbw_{tf}'].pct_change() * 100
            
            df_raw = pd.merge_asof(df_raw.sort_values('ts'), 
                                   df_tf[['ts', f'adx_{tf}', f'adx_slope_{tf}', f'atr_{tf}', f'rsi_{tf}', 
                                          f'vol_{tf}_mean', f'bbw_{tf}', f'bbw_slope_{tf}']].sort_values('ts'), 
                                   on='ts', direction='backward')
            df_raw[f'cum_vol_{tf}'] = df_raw.groupby(df_raw['ts'].dt.floor(tf.lower().replace('m', 'min')))['vol'].transform('cumsum')
    return df_raw.dropna().reset_index(drop=True)

# [3. 백테스트 엔진]
def evaluate(args):
    ind_vals, df_main = args
    ind = ind_vals if isinstance(ind_vals, dict) else dict(zip(GENE_BOUNDS.keys(), ind_vals))
    r_tf, tn_tf, ts_tf, atr_tf = ind.get('r_inter', '1h'), ind.get('t_inter_normal', '2h'), ind.get('t_inter_strong', '1h'), ind.get('atr_inter', '4h')
    
    bal, peak, mdd = 100.0, 100.0, 0.0
    pos = None
    pos_duration = 0 # 보유 시간 카운트 변수 추가
    stats = {'range': {'wins': 0, 'trades': 0}, 'trend_normal': {'wins': 0, 'trades': 0}, 'trend_strong': {'wins': 0, 'trades': 0}, 'gross_p': 0.0, 'gross_l': 1e-9}

    for row in df_main.itertuples():
        curr_p = row.close
        if pos is None:
            side, mode = None, None
            # 1. Range 진입 판정 (BBW Slope 조건 포함)
            if getattr(row, f"adx_{r_tf}") < ind['r_adx_limit'] and \
               getattr(row, f"adx_slope_{r_tf}") <= ind['r_slope_max'] and \
               getattr(row, f"bbw_slope_{r_tf}") < 0:
                if row.vol > (row.vol_mean * ind['r_vol_limit']):
                    rsi_v, ma_v = getattr(row, f"rsi_{r_tf}"), getattr(row, f"ma20_{r_tf}")
                    side = 'long' if (rsi_v < ind['rsi_low'] and curr_p < ma_v) else \
                           'short' if (rsi_v > ind['rsi_high'] and curr_p > ma_v) else None
                    if side: mode = 'range'
            
            # 2. Trend 진입 판정
            if not mode:
                if getattr(row, f"adx_{ts_tf}") > ind['t_adx_limit_strong'] and getattr(row, f"adx_slope_{ts_tf}") >= ind['t_slope_strong']:
                    if getattr(row, f"cum_vol_{ts_tf}") > (getattr(row, f"vol_{ts_tf}_mean") * ind['t_vol_limit_strong']):
                        rsi_v, ma_v = getattr(row, f"rsi_{ts_tf}"), getattr(row, f"ma20_{ts_tf}")
                        side = 'long' if (curr_p > ma_v and rsi_v < ind['t_rsi_max_strong']) else \
                               'short' if (curr_p < ma_v and rsi_v > ind['t_rsi_min_strong']) else None
                        if side: mode = 'trend_strong'
                if not mode:
                    if getattr(row, f"adx_{tn_tf}") > ind['t_adx_limit_normal'] and getattr(row, f"adx_slope_{tn_tf}") >= ind['t_slope_min']:
                        if getattr(row, f"cum_vol_{tn_tf}") > (getattr(row, f"vol_{tn_tf}_mean") * ind['t_vol_limit_normal']):
                            rsi_v, ma_v = getattr(row, f"rsi_{tn_tf}"), getattr(row, f"ma20_{tn_tf}")
                            side = 'long' if (curr_p > ma_v and rsi_v < ind['t_rsi_max_normal']) else \
                                   'short' if (curr_p < ma_v and rsi_v > ind['t_rsi_min_normal']) else None
                            if side: mode = 'trend_normal'

            if mode and side:
                atr_pct = getattr(row, f"atr_{atr_tf}") / (curr_p + 1e-9)
                pos_duration = 0 # 진입 시 초기화
                
                if mode == 'range':
                    tp_pct = atr_pct * ind['r_tp_mult']
                    # [수비 포인트 1] 횡보 손절 하드캡 2% 적용
                    sl_pct = min(ind['r_sl_mult'] / (atr_pct + 1e-9), 0.02)
                else:
                    sl_base = ind['t_sl_base_strong'] if mode == 'trend_strong' else ind['t_sl_base_normal']
                    tp_mult = ind['t_tp_mult'] if mode == 'trend_strong' else ind['t_tp_short_mult']
                    tp_pct = atr_pct * tp_mult
                    sl_pct = min(sl_base / (atr_pct + 1e-9), 0.05)

                if side == 'long':
                    tp, sl = curr_p * (1 + tp_pct), curr_p * (1 - sl_pct)
                else:
                    tp, sl = curr_p * (1 - tp_pct), curr_p * (1 + sl_pct)
                pos = {'side': side, 'ent_p': curr_p, 'sl': sl, 'tp': tp, 'mode': mode}

        else:
            pos_duration += 1
            is_exit = False
            exit_p = curr_p
            
            # [수비 포인트 2] 횡보 모드 타임아웃 (15봉)
            if pos['mode'] == 'range' and pos_duration >= 15:
                is_exit = True
                exit_p = curr_p
            
            # 익손절 체크
            if not is_exit:
                if (curr_p <= pos['sl'] if pos['side'] == 'long' else curr_p >= pos['sl']):
                    is_exit = True
                    exit_p = pos['sl']
                elif (curr_p >= pos['tp'] if pos['side'] == 'long' else curr_p <= pos['tp']):
                    is_exit = True
                    exit_p = pos['tp']

            if is_exit:
                # [수비 포인트 3] 타임아웃/강제종료 시 슬리피지 감안 수수료 계산
                pnl = bal * (((exit_p - pos['ent_p'])/pos['ent_p'] if pos['side'] == 'long' else (pos['ent_p'] - exit_p)/pos['ent_p']) - (TAKER_FEE*2 + SLIPPAGE)) * LEVERAGE
                bal += pnl
                stats[pos['mode']]['trades'] += 1
                if pnl > 0: stats[pos['mode']]['wins'] += 1; stats['gross_p'] += pnl
                else: stats['gross_l'] += abs(pnl)
                
                pos = None
                if bal > peak: peak = bal
                mdd = max(mdd, (peak - bal) / (peak + 1e-9))
                if mdd > MDD_LIMIT or bal <= 5.0: break

    # 평가 지표 산출
    r_tr, n_tr, s_tr = stats['range']['trades'], stats['trend_normal']['trades'], stats['trend_strong']['trades']
    r_wr = stats['range']['wins'] / r_tr if r_tr > 0 else 0
    n_wr = stats['trend_normal']['wins'] / n_tr if n_tr > 0 else 0
    s_wr = stats['trend_strong']['wins'] / s_tr if s_tr > 0 else 0

    if (r_tr < 3) or (r_wr == 1.0) or (n_tr < 3) or (n_wr == 1.0) or (s_tr < 3) or (s_wr == 1.0) or (mdd > MDD_LIMIT):
        return {'Fitness': -1000000.0, 'ROI': bal - 100, 'PF': 0, 'MDD': mdd, 'Trades': r_tr + n_tr + s_tr, **ind}

    pf = (stats['gross_p'] / stats['gross_l']) if stats['gross_l'] > 0 else 1.0
    fitness = (bal - 100) * pf if bal > 0 else -100000.0
    return {'Fitness': fitness, 'ROI': bal - 100, 'PF': pf, 'MDD': mdd, 'Trades': r_tr + n_tr + s_tr, 
            'R_Tr': r_tr, 'TN_Tr': n_tr, 'TS_Tr': s_tr, **ind}

# [4. GA 메인 루프]
def run_ga(df_train):
    population = [{**{k: random.uniform(v[0], v[1]) for k, v in GENE_BOUNDS.items()}, 
                   **{tf: random.choice(INTERVALS) for tf in TF_KEYS}} for _ in range(POP_SIZE)]
    best_overall_fitness, no_improvement_count, best_overall_result = -float('inf'), 0, None
    print(f"🧬 GA 최적화 시작: POP={POP_SIZE}, SYMBOL={SYMBOL}")

    for gen in range(GENERATIONS):
        with Pool(cpu_count()) as p:
            results = p.map(evaluate, [(ind, df_train) for ind in population])
        results.sort(key=lambda x: x['Fitness'], reverse=True)
        current_best = results[0]

        if current_best['Fitness'] > best_overall_fitness:
            best_overall_fitness, best_overall_result, no_improvement_count = current_best['Fitness'], current_best, 0
        else: no_improvement_count += 1

        print(f" 세대 {gen+1:3d} | Fitness: {current_best['Fitness']:.2f} | ROI: {current_best['ROI']:.2f}% | MDD: {current_best['MDD']*100:.1f}% | 정체: {no_improvement_count}/{PATIENCE}")
        if no_improvement_count >= PATIENCE: break

        elites = results[:ELITE_SIZE]
        new_pop = [{k: v for k, v in e.items() if k in list(GENE_BOUNDS.keys()) + TF_KEYS} for e in elites]
        while len(new_pop) < POP_SIZE:
            p1, p2 = random.sample(elites, 2)
            child = {k: (random.uniform(GENE_BOUNDS[k][0], GENE_BOUNDS[k][1]) if random.random() < MUTATION_RATE else random.choice([p1[k], p2[k]])) for k in GENE_BOUNDS.keys()}
            for tf in TF_KEYS: child[tf] = random.choice([p1[tf], p2[tf]])
            new_pop.append(child)
        population = new_pop
    return best_overall_result

if __name__ == "__main__":
    df_all = prepare_full_data()
    if df_all is not None:
        print(f"\n🚀 {SYMBOL} 딥-최적화 시작 (New Bounds & Inverse SL 적용)")
        best_params = run_ga(df_all)
        print("\n" + "="*50)
        print("🏆 최적 파라미터 결과")
        pprint.pprint(best_params)
        pd.DataFrame([best_params]).to_csv("Final_Optimized_Params.csv", index=False, encoding='utf-8-sig')
        print("="*50)