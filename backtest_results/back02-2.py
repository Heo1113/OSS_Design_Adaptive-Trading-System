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

GENE_BOUNDS = {
    'r_adx_limit': (15.0, 35.0), 
    't_adx_limit_normal': (20.0, 50.0), 
    't_adx_limit_strong': (35.0, 65.0), 
    'r_slope_max': (-5.0, 0.0),
    't_slope_min': (1.0, 15.0),    
    't_slope_strong': (3.0, 20.0),   
    't_tp_short_mult': (1.0, 3.0),
    't_tp_mult': (4.0, 15.0),     
    'r_tp_mult': (1.5, 5.0),      
    'r_sl_mult': (0.5, 2.5),      
    't_sl_base_normal': (1.5, 4.0),
    't_sl_base_strong': (1.0, 3.0),
    't_ts_mult': (1.0, 3.0),      
    't_sl_activate': (0.01, 0.06), 
    'r_vol_limit': (0.5, 2.5),    
    't_vol_limit_normal': (0.5, 2.5),
    't_vol_limit_strong': (1.0, 4.0),
    'rsi_low': (25.0, 45.0),      
    'rsi_high': (55.0, 75.0),     
    't_rsi_max_normal': (60.0, 85.0),
    't_rsi_min_normal': (15.0, 40.0),
    't_rsi_max_strong': (70.0, 95.0),
    't_rsi_min_strong': (5.0, 30.0)
}

INTERVALS = ['1h', '2h', '4h']
TF_KEYS = ['r_inter', 't_inter_normal', 't_inter_strong', 'atr_inter']

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
    print(f"🔄 {SYMBOL} 데이터 로딩 (전체 구간 및 지표 계산)...")
    df_raw = get_data(SYMBOL, '3m', TOTAL_DAYS + BUFFER_DAYS)
    if df_raw.empty: return None
    df_raw['vol_mean'] = df_raw['vol'].rolling(20).mean()
    
    for tf in INTERVALS:
        print(f"   ∟ {tf} 타임프레임 지표 결합 중...")
        df_tf = get_data(SYMBOL, tf, TOTAL_DAYS + BUFFER_DAYS)
        if not df_tf.empty:
            df_tf[f'ma20_{tf}'] = ta.sma(df_tf['close'], length=20)
            adx_res = ta.adx(df_tf['high'], df_tf['low'], df_tf['close'])
            df_tf[f'adx_{tf}'] = adx_res['ADX_14']
            df_tf[f'adx_slope_{tf}'] = adx_res['ADX_14'].pct_change() * 100
            df_tf[f'atr_{tf}'] = ta.atr(df_tf['high'], df_tf['low'], df_tf['close'], length=14)
            df_tf[f'rsi_{tf}'] = ta.rsi(df_tf['close'], length=14)
            df_tf[f'vol_{tf}_mean'] = df_tf['vol'].rolling(20).mean()
            bb = ta.bbands(df_tf['close'], length=20, std=2.0)
            df_tf[f'bbw_{tf}'] = (bb.iloc[:, 2] - bb.iloc[:, 0]) / (bb.iloc[:, 1] + 1e-9)
            df_tf[f'bbw_slope_{tf}'] = df_tf[f'bbw_{tf}'].pct_change() * 100
            
            df_raw = pd.merge_asof(df_raw.sort_values('ts'), df_tf[['ts', f'ma20_{tf}', f'adx_{tf}', f'adx_slope_{tf}', f'atr_{tf}', f'rsi_{tf}', f'vol_{tf}_mean', f'bbw_{tf}', f'bbw_slope_{tf}']].sort_values('ts'), on='ts', direction='backward')
            df_raw[f'cum_vol_{tf}'] = df_raw.groupby(df_raw['ts'].dt.floor(tf.lower().replace('m', 'min')))['vol'].transform('cumsum')
    
    return df_raw.dropna().reset_index(drop=True)

def evaluate(args):
    ind_vals, df_main = args
    ind = ind_vals if isinstance(ind_vals, dict) else dict(zip(list(GENE_BOUNDS.keys()) + TF_KEYS, ind_vals))
    
    r_tf, tn_tf, ts_tf, atr_tf = ind['r_inter'], ind['t_inter_normal'], ind['t_inter_strong'], ind['atr_inter']
    
    bal, peak, mdd, pos, pos_duration = 100.0, 100.0, 0.0, None, 0
    trade_returns = [] 
    stats = {'range': {'trades': 0, 'wins': 0}, 'trend_n': {'trades': 0, 'wins': 0}, 'trend_s': {'trades': 0, 'wins': 0}, 'gross_p': 0.0, 'gross_l': 1e-9}

    for row in df_main.itertuples():
        curr_p = row.close
        if pos is None:
            side, mode = None, None
            # 횡보
            if getattr(row, f"adx_{r_tf}") < ind['r_adx_limit'] and getattr(row, f"bbw_slope_{r_tf}") < ind['r_slope_max']:
                if row.vol > (row.vol_mean * ind['r_vol_limit']):
                    rsi_v, ma_v = getattr(row, f"rsi_{r_tf}"), getattr(row, f"ma20_{r_tf}")
                    if rsi_v < ind['rsi_low'] and curr_p < ma_v: side, mode = 'long', 'range'
                    elif rsi_v > ind['rsi_high'] and curr_p > ma_v: side, mode = 'short', 'range'
            
            # 추세 Normal
            if not mode:
                if getattr(row, f"adx_{tn_tf}") > ind['t_adx_limit_normal'] and getattr(row, f"adx_slope_{tn_tf}") >= ind['t_slope_min']:
                    if getattr(row, f"cum_vol_{tn_tf}") > (getattr(row, f"vol_{tn_tf}_mean") * ind['t_vol_limit_normal']):
                        rsi_v, ma_v = getattr(row, f"rsi_{tn_tf}"), getattr(row, f"ma20_{tn_tf}")
                        if ind['t_rsi_min_normal'] < rsi_v < ind['t_rsi_max_normal']:
                            if curr_p > ma_v: side, mode = 'long', 'trend_n'
                            else: side, mode = 'short', 'trend_n'
            
            # 추세 Strong
            if not mode:
                if getattr(row, f"adx_{ts_tf}") > ind['t_adx_limit_strong'] and getattr(row, f"adx_slope_{ts_tf}") >= ind['t_slope_strong']:
                    if getattr(row, f"cum_vol_{ts_tf}") > (getattr(row, f"vol_{ts_tf}_mean") * ind['t_vol_limit_strong']):
                        rsi_v, ma_v = getattr(row, f"rsi_{ts_tf}"), getattr(row, f"ma20_{ts_tf}")
                        if ind['t_rsi_min_strong'] < rsi_v < ind['t_rsi_max_strong']:
                            if curr_p > ma_v: side, mode = 'long', 'trend_s'
                            else: side, mode = 'short', 'trend_s'
            
            if mode and side:
                atr_val = getattr(row, f"atr_{atr_tf}")
                pos_duration = 0
                if mode == 'range':
                    tp_dist, sl_dist = atr_val * ind['r_tp_mult'], atr_val * ind['r_sl_mult']
                elif mode == 'trend_n':
                    tp_dist, sl_dist = atr_val * ind['t_tp_mult'], atr_val * ind['t_sl_base_normal']
                else:
                    tp_dist, sl_dist = atr_val * (ind['t_tp_mult'] * ind['t_tp_short_mult']), atr_val * ind['t_sl_base_strong']
                
                tp = (curr_p + tp_dist) if side == 'long' else (curr_p - tp_dist)
                sl = (curr_p - sl_dist) if side == 'long' else (curr_p + sl_dist)
                pos = {'side': side, 'ent_p': curr_p, 'sl': sl, 'tp': tp, 'mode': mode}
        
        else:
            pos_duration += 1
            is_exit, exit_p = False, curr_p
            if 'trend' in pos['mode']:
                roi_curr = (curr_p - pos['ent_p'])/pos['ent_p'] if pos['side'] == 'long' else (pos['ent_p'] - curr_p)/pos['ent_p']
                if roi_curr > ind['t_sl_activate']:
                    atr_v = getattr(row, f"atr_{atr_tf}")
                    if pos['side'] == 'long': pos['sl'] = max(pos['sl'], curr_p - atr_v * ind['t_ts_mult'])
                    else: pos['sl'] = min(pos['sl'], curr_p + atr_v * ind['t_ts_mult'])

            if pos['mode'] == 'range' and pos_duration >= 15: is_exit = True
            elif (curr_p <= pos['sl'] if pos['side'] == 'long' else curr_p >= pos['sl']): is_exit, exit_p = True, pos['sl']
            elif (curr_p >= pos['tp'] if pos['side'] == 'long' else curr_p <= pos['tp']): is_exit, exit_p = True, pos['tp']
            
            if is_exit:
                pnl = bal * (((exit_p - pos['ent_p'])/pos['ent_p'] if pos['side'] == 'long' else (pos['ent_p'] - exit_p)/pos['ent_p']) - (TAKER_FEE*2 + SLIPPAGE*2)) * LEVERAGE
                bal += pnl
                trade_returns.append(pnl)
                stats[pos['mode']]['trades'] += 1
                if pnl > 0: stats[pos['mode']]['wins'] += 1; stats['gross_p'] += pnl
                else: stats['gross_l'] += abs(pnl)
                pos = None
                if bal > peak: peak = bal
                mdd = max(mdd, (peak - bal) / (peak + 1e-9))
                if mdd > MDD_LIMIT or bal <= 5.0: break

    tr_cnt = len(trade_returns)
    if tr_cnt < 5 or mdd > MDD_LIMIT: return {'Fitness': -1e6, **ind}
    
    roi = bal - 100
    calmar = roi / (mdd * 100 + 1e-9)
    if len(trade_returns) >= 2:
        consistency = np.mean(trade_returns) / (np.std(trade_returns) + 1e-9)
    else:
        consistency = 0.0
        
    pf = stats['gross_p'] / stats['gross_l']
    fitness = roi * pf * (1 + consistency)
    return {'Fitness': fitness, 'ROI': roi, 'PF': pf, 'MDD': mdd, 'Calmar': calmar, 'Consistency': consistency, 'Trades': tr_cnt, **ind}

def run_ga(df_train):
    population = [{**{k: random.uniform(v[0], v[1]) for k, v in GENE_BOUNDS.items()}, **{tf: random.choice(INTERVALS) for tf in TF_KEYS}} for _ in range(POP_SIZE)]
    best_overall_fitness = -1e9
    best_overall_result = None
    no_improvement_count = 0
    
    for gen in range(GENERATIONS):
        with Pool(cpu_count()) as p:
            results = p.map(evaluate, [(ind, df_train) for ind in population])
        results = [r for r in results if r['Fitness'] > -1e5]
        if not results:
            population = [{**{k: random.uniform(v[0], v[1]) for k, v in GENE_BOUNDS.items()}, **{tf: random.choice(INTERVALS) for tf in TF_KEYS}} for _ in range(POP_SIZE)]
            continue
            
        results.sort(key=lambda x: x['Fitness'], reverse=True)
        current_best = results[0]
        
        if current_best['Fitness'] > best_overall_fitness:
            best_overall_fitness, best_overall_result, no_improvement_count = current_best['Fitness'], current_best, 0
        else: no_improvement_count += 1

        print(f" 세대 {gen+1:3d} | Fitness: {current_best['Fitness']:.4f} | ROI: {current_best['ROI']:.2f}% | "
              f"MDD: {current_best['MDD']*100:.1f}% | Calmar: {current_best.get('Calmar', 0):.3f} | "
              f"Consistency: {current_best.get('Consistency', 0):.3f} | 정체: {no_improvement_count}/{PATIENCE}")
        
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
        best_res = run_ga(df_all)
        print("\n" + "="*50)
        print("🏆 최적 파라미터 검색 완료")
        pprint.pprint(best_res)
