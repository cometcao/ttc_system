import numpy as np
import pandas as pd
import talib
from prettytable import PrettyTable
import types
from common_include import *
from oop_strategy_frame import *
from oop_adjust_pos import *
from oop_stop_loss import *
from oop_select_stock import *
from oop_sort_stock import *
from oop_record_stats import *
from ML_main import *
import traceback
import sys
# from oop_trading_sync import *


# 不同步的白名单，主要用于实盘易同步持仓时，不同步中的新股，需把新股代码添加到这里。https://www.joinquant.com/algorithm/index/edit?algorithmId=23c589f4594f827184d4f6f01a11b2f2
# 可把white_list另外放到研究的一个py文件里
def white_list():
    return ['000001.XSHE']

# ==================================策略配置==============================================
def select_strategy(context):
    context.strategy_memo = '混合策略'
    # **** 这里定义log输出的类类型,重要，一定要写。假如有需要自定义log，可更改这个变量
    context.log_type = Rule_loger
    # 判断是运行回测还是运行模拟
    context.is_sim_trade = context.run_info.run_type == RUN_TYPE.PAPER_TRADING
    context.port_pos_control = 1.0 # 组合仓位控制参数
    context.monitor_levels = ['5d','1d','60m']
    context.buy_count = 3
    context.pb_limit = 5
    context.ps_limit = 2.5
    context.pe_limit = 200
    context.evs_limit = 5
    context.eve_limit = 5
    index2 = '000016.XSHG'  # 大盘指数
    index8 = '399333.XSHE'  # 小盘指数
    context.money_fund = ['511880.XSHG','511010.XSHG','511220.XSHG']
    
    ''' ---------------------配置 调仓条件判断规则-----------------------'''
    # 调仓条件判断
    adjust_condition_config = [
        [True, '_time_c_', '调仓时间', Time_condition, {
            # 'times': [[10,0],[10, 30], [11,00], [13,00], [13,30], [14,00],[14, 30]],  # 调仓时间列表，二维数组，可指定多个时间点
            'times': [[10, 30], [11,20], [13,30], [14, 50]],  # 调仓时间列表，二维数组，可指定多个时间点
        }],
        [False, '_Stop_loss_by_price_', '指数最高低价比值止损器', Stop_loss_by_price, {
            'index': '000001.XSHG',  # 使用的指数,默认 '000001.XSHG'
            'day_count': 160,  # 可选 取day_count天内的最高价，最低价。默认160
            'multiple': 2.2  # 可选 最高价为最低价的multiple倍时，触 发清仓
        }],
        [False,'_Stop_loss_by_3_black_crows_','指数三乌鸦止损', Stop_loss_by_3_black_crows,{
            'index':'000001.XSHG',  # 使用的指数,默认 '000001.XSHG'
             'dst_drop_minute_count':60,  # 可选，在三乌鸦触发情况下，一天之内有多少分钟涨幅<0,则触发止损，默认60分钟
            }],
        [False,'Stop_loss_stocks','个股止损',Stop_gain_loss_stocks,{
            'period':20,  # 调仓频率，日
            'stop_loss':0.0,
            'enable_stop_loss':True,
            'stop_gain':0.2,
            'enable_stop_gain':False
            },],
        # [True,'_Stop_loss_by_growth_rate_','当日指数涨幅止损器',Stop_loss_by_growth_rate,{
        #     'index':'000001.XSHG',  # 使用的指数,默认 '000001.XSHG'
        #      'stop_loss_growth_rate':-0.05,
        #     }],
        # [False,'_Stop_loss_by_28_index_','28实时止损',Stop_loss_by_28_index,{
        #             'index2' : '000016.XSHG',       # 大盘指数
        #             'index8' : '399333.XSHE',       # 小盘指数
        #             'index_growth_rate': 0.01,      # 判定调仓的二八指数20日增幅
        #             'dst_minute_count_28index_drop': 120 # 符合条件连续多少分钟则清仓
        #         }],
        [False, '_equity_curve_protect_', '资金曲线止损', equity_curve_protect, {
            'day_count': 20,  # 
            'percent': 0.01,  # 可选 当日总资产小于选定日前资产的固定百分数，触 发清仓
            'use_avg': False,
            'market_index':'000300.XSHG'
        }],
        [False, '', '多指数20日涨幅止损器', Mul_index_stop_loss, {
            'indexs': [index2, index8],
            'min_rate': 0.005
        }],
        [False, '', '多指数技术分析止损器', Mul_index_stop_loss_ta, {
            'indexs': [index2, index8],
            'ta_type': TaType.TRIX_PURE,
            'period':'5d'
        }],
        [True, '', '调仓日计数器', Period_condition, {
            'period': 1,  # 调仓频率,日
        }],
    ]
    adjust_condition_config = [
        [True, '_adjust_condition_', '调仓执行条件的判断规则组合', Group_rules, {
            'config': adjust_condition_config
        }]
    ]

    ''' --------------------------配置 选股规则----------------- '''
    pick_config = [
        [True, '', '缠论强弱势板块', Pick_rank_sector,{
                        'strong_sector':True, 
                        'sector_limit_pct': 13,
                        'strength_threthold': 0, 
                        'isDaily': False, 
                        'useIntradayData':False,
                        'useAvg':False,
                        'avgPeriod':5}],
        [True, '', '基本面数据筛选', Filter_financial_data2, {
            'factors': [
                        FD_Factor('fundamentals.eod_derivative_indicator.ps_ratio', min=0, max=context.ps_limit),
                        FD_Factor('fundamentals.eod_derivative_indicator.pe_ratio', min=0, max=context.pe_limit),
                        # FD_Factor('fundamentals.eod_derivative_indicator.pcf_ratio', min=0, max=100),
                        FD_Factor('fundamentals.eod_derivative_indicator.pb_ratio', min=0, max=context.pb_limit),
                        # FD_Factor(eve_query_string
                        #         , min=0
                        #         , max=5
                        #         , isComplex=True),
                        # FD_Factor(evs_query_string
                        #         , min=0
                        #         , max=5
                        #         , isComplex=True),
                        ],
            'order_by': 'fundamentals.eod_derivative_indicator.market_cap',
            'sort': SortType.asc,
            'by_sector': False,
            'limit':50}],
        [False, '', '小佛雪选股', Filter_FX_data, {'limit':50}],
        # 测试的多因子选股,所选因子只作为示例。
        # 选用的财务数据参考 https://www.joinquant.com/data/dict/fundamentals
        # 传入参数的财务因子需为字符串，原因是直接传入如 indicator.eps 会存在序列化问题。
        # FD_Factor 第一个参数为因子，min=为最小值 max为最大值，=None则不限，默认都为None。min,max都写则为区间
        [True, '', '过滤创业板', Filter_gem, {}],
        [True, '', '过滤ST,停牌,涨跌停股票', Filter_common, {}],
        [False, '', '强势股筛选', Filter_Herd_head_stocks,{'gainThre':0.05, 'count':20, 'useIntraday':True, 'filter_out':True}],
        [False, '', '技术分析筛选-AND', checkTAIndicator_AND, { 
            'TA_Indicators':[
                            # (TaType.BOLL,'5d',233),
                            (TaType.TRIX_STATUS, '5d', 100),
                            # (TaType.MACD_ZERO, '5d', 100),
                            (TaType.MA, '1d', 20),
                            # (TaType.MA, '1d', 60),
                            ],
            'isLong':True}], # 确保大周期安全
        [False, '', '技术分析筛选-OR', checkTAIndicator_OR, { 
            'TA_Indicators':[
                            # (TaType.BOLL,'5d',233),
                            (TaType.TRIX_STATUS, '5d', 100),
                            # (TaType.MACD_STATUS, '5d', 100),
                            # (TaType.MA, '1d', 20),
                            # (TaType.MA, '1d', 60),
                            ],
            'isLong':True}], # 确保大周期安全
        [False, '', '日线周线级别表里买点筛选', Filter_Week_Day_Long_Pivot_Stocks, {'monitor_levels':context.monitor_levels}],
        [True, '', '权重排序', SortRules, {
            'config': [
                [True, 'Sort_std_data', '波动率排序', Sort_std_data, {
                    'sort': SortType.asc
                    , 'period': 60
                    , 'weight': 100}],
                [False, 'cash_flow_rank', '庄股脉冲排序', Sort_cash_flow_rank, {
                    'sort': SortType.desc
                    , 'weight': 100}],
                [False, '', '市值排序', Sort_financial_data, {
                    'factor': 'fundamentals.eod_derivative_indicator.market_cap',
                    'sort': SortType.asc
                    , 'weight': 100}],
                [False, '', 'EVS排序', Sort_financial_data, {
                    'factor': evs_query_string,
                    'sort': SortType.asc
                    , 'weight': 100}],
                [False, '', '流通市值排序', Sort_financial_data, {
                    'factor': 'fundamentals.eod_derivative_indicator.a_share_market_val_2',
                    'sort': SortType.asc
                    , 'weight': 100}],
                [False, '', 'P/S排序', Sort_financial_data, {
                    'factor': 'fundamentals.eod_derivative_indicator.ps_ratio',
                    'sort': SortType.asc
                    , 'weight': 100}],
                [False, '', 'GP排序', Sort_financial_data, {
                    'factor': 'income.total_profit/balance.total_assets',
                    'sort': SortType.desc
                    , 'weight': 100}],
                [False, '', '按当前价排序', Sort_price, {
                    'sort': SortType.asc
                    , 'weight': 20}],
                [False, '5growth', '5日涨幅排序', Sort_growth_rate, {
                    'sort': SortType.asc
                    , 'weight': 100
                    , 'day': 5}],
                [False, '20growth', '20日涨幅排序', Sort_growth_rate, {
                    'sort': SortType.asc
                    , 'weight': 100
                    , 'day': 20}],
                [False, '60growth', '60日涨幅排序', Sort_growth_rate, {
                    'sort': SortType.asc
                    , 'weight': 10
                    , 'day': 60}],
                [False, 'cash_flow_rank', '庄股脉冲排序', Sort_cash_flow_rank, {
                    'sort': SortType.desc
                    , 'weight': 100}],
                [False, '', '按换手率排序', Sort_turnover_ratio, {
                    'sort': SortType.desc
                    , 'weight': 50}],
            ]}
        ],
        [True, '', '获取最终选股数', Filter_buy_count, {
            'buy_count': 20  # 最终入选股票数
        }],
    ]
    pick_new = [
        [True, '_pick_stocks_', '选股', Pick_stocks2, {
            'config': pick_config,
            'day_only_run_one': True
        }]
    ]

    ''' --------------------------配置 4 调仓规则------------------ '''
    # # 通达信持仓字段不同名校正
    col_names = {'可用': u'可用', '市值': u'参考市值', '证券名称': u'证券名称', '资产': u'资产'
        , '证券代码': u'证券代码', '证券数量': u'证券数量', '可卖数量': u'可卖数量', '当前价': u'当前价', '成本价': u'成本价'
                 }
    adjust_position_config = [
        [False, '', '卖出股票', Sell_stocks, {}],
        [False, '', '买入股票', Buy_stocks, {
            'buy_count': context.buy_count  # 最终买入股票数
        }],
        [True, '', '卖出股票日内表里', Sell_stocks_chan, {'monitor_levels': context.monitor_levels}],
        [False, '', '买入股票日内表里', Buy_stocks_chan, {
            'buy_count': context.buy_count,
            'monitor_levels': context.monitor_levels, 
            'pos_control':context.port_pos_control}],
        [True,'','VaR方式买入股票', Buy_stocks_var, {
            'buy_count': context.buy_count,
            'monitor_levels': context.monitor_levels, 
            'pos_control':context.port_pos_control,
            'money_fund':context.money_fund,
            'adjust_pos':True,
            'equal_pos':True,
            }],
        [True, '_Show_postion_adjust_', '显示买卖的股票', Show_postion_adjust, {}],
        # [context.is_sim_trade,'trade_Xq','Xue Qiu Webtrader',XueQiu_order,{'version':3}],
    ]
    adjust_position_config = [
        [True, '_Adjust_position_', '调仓执行规则组合', Adjust_position, {
            'config': adjust_position_config
        }]
    ]

    ''' --------------------------配置 辅助规则------------------ '''
    # 优先辅助规则，每分钟优先执行handle_data
    common_config_list = [
        [True, '', '设置系统参数', Set_sys_params, {
            'benchmark': '000300.XSHG'  # 指定基准为次新股指
        }],
        [True, '', '持仓信息打印器', Show_position, {}],
        [True, '', '统计执行器', Stat, {'trade_stats':False}],
        [False, '', '自动调参器', Update_Params_Auto, {
            'ps_threthold':0.618,
            'pb_threthold':0.618,
            'pe_threthold':0.809,
            'buy_threthold':0.809,
            'evs_threthold':0.618,
            'eve_threthold':0.618,
            'pos_control_value': 1.0
        }],
        # [context.is_sim_trade, '_Purchase_new_stocks_', '实盘易申购新股', Purchase_new_stocks, {
        #     'times': [[11, 24]],
        #     'host':'111.111.111.111',   # 实盘易IP
        #     'port':8888,    # 实盘易端口
        #     'key':'',   # 实盘易Key
        #     'clients': ['title:zhaoshang', 'title:guolian'] # 实盘易client列表,即一个规则支持同一个实盘易下的多个帐号同时打新
        # }],
    ]
    common_config = [
        [True, '_other_pre_', '预先处理的辅助规则', Group_rules, {
            'config': common_config_list
        }]
    ]
    # 组合成一个总的策略
    context.main_config = (common_config
                     + adjust_condition_config
                     + pick_new
                     + adjust_position_config)




# 在这个方法中编写任何的初始化逻辑。context对象将会在你的算法策略的任何方法之间做传递。
def init(context):
    # 策略配置
    select_strategy(context)
    # 创建策略组合
    context.main = Strategy_Group({'config': context.main_config
                                , 'g_class': Global_variable
                                , 'memo': context.strategy_memo
                                , 'name': '_main_'})
    context.main.initialize(context)
    # 打印规则参数
    context.main.log.info(context.main.show_strategy())


# 你选择的证券的数据更新将会触发此段逻辑，例如日或分钟历史数据切片或者是实时数据切片更新
# 按分钟回测
def handle_bar(context, bar_dict):
    # 保存context到全局变量量，主要是为了方便规则器在一些没有context的参数的函数里使用。
    context.main.g.context = context
    # 执行策略
    context.main.handle_data(context, bar_dict)


# 开盘
def before_trading(context):
    logger.info("==========================================================================")
    context.main.g.context = context
    try:
        context.main.process_initialize(context)
    except:
        pass
    context.main.before_trading_start(context)


# 收盘
def after_trading(context):
    context.main.g.context = context
    context.main.after_trading_end(context)
    context.main.g.context = None
    
    
    
# ''' ----------------------参数自动调整----------------------------'''
class Update_Params_Auto(Rule):
    def __init__(self, params):
        Rule.__init__(self, params)
        self.ps_threthold = params.get('ps_threthold',0.8)
        self.pb_threthold = params.get('pb_threthold',0.618)
        self.pe_threthold = params.get('pe_threthold',0.8)
        self.evs_threthold = params.get('evs_threthold',0.618)
        self.eve_threthold = params.get('eve_threthold',0.618)
        self.buy_threthold = params.get('buy_threthold', 0.9)
        self.pos_control_value = params.get('pos_control_value', 0.5)

    def before_trading_start(self, context):
        if self.g.isFirstTradingDayOfWeek(context):
            context.ps_limit = self.g.getFundamentalThrethold('fundamentals.eod_derivative_indicator.ps_ratio', self.ps_threthold)
            context.pb_limit = self.g.getFundamentalThrethold('fundamentals.eod_derivative_indicator.pb_ratio', self.pb_threthold)
            context.pe_limit = self.g.getFundamentalThrethold('fundamentals.eod_derivative_indicator.pe_ratio', self.pe_threthold)
            # context.evs_limit = self.g.getFundamentalThrethold(evs_query_string, self.evs_threthold)
            # context.eve_limit = self.g.getFundamentalThrethold(eve_query_string, self.eve_threthold)
            
            self.dynamicBuyCount(context)
            self.log.info("每周修改全局参数: ps_limit: %s pb_limit: %s pe_limit: %s buy_count: %s evs_limit: %s eve_limit: %s" % (context.ps_limit, context.pb_limit, context.pe_limit, context.buy_count, context.evs_limit, context.eve_limit))
        
        # self.doubleIndexControl('000016.XSHG', '399333.XSHE')
        # self.log.info("每日修改全局参数: port_pos_control: %s" % (g.port_pos_control))
        self.updateRelaventRules(context)
    
    def dynamicBuyCount(self, context):
        import math
        context.buy_count = int(math.ceil(math.log(context.portfolio.total_value/10000)))
        
    def doubleIndexControl(self, index1, index2, target=0, period=20):
        gr_index1 = get_growth_rate(index1, period)
        gr_index2 = get_growth_rate(index2, period)
        target = 0.01
        if gr_index1 < target and gr_index2 < target:
            context.port_pos_control = self.pos_control_value / 2
        elif gr_index1 >= target or gr_index2 >= target:
            context.port_pos_control = self.pos_control_value
        else:
            context.port_pos_control = 1.0  

    def updateRelaventRules(self, context):
        # print g.main.rules
        # update everything
        for rule in g.main.rules:
            if isinstance(rule, Pick_stocks2):
                for r2 in rule.rules:
                    if isinstance(r2, Filter_financial_data2):
                        r2.update_params(context, {
                            'factors': [
                                        FD_Factor('fundamentals.eod_derivative_indicator.ps_ratio', min=0, max=context.ps_limit),
                                        FD_Factor('fundamentals.eod_derivative_indicator.pe_ratio', min=0, max=context.pe_limit),
                                        FD_Factor('fundamentals.eod_derivative_indicator.pb_ratio', min=0, max=context.pb_limit),
                                        # FD_Factor(evs_query_string, min=0, max=context.evs_limit, isComplex=True),
                                        # FD_Factor(eve_query_string, min=0, max=context.eve_limit, isComplex=True),
                                        ],
                            'order_by': 'eod_derivative_indicator.a_share_market_val_2',
                            'sort': SortType.asc,
                            'by_sector':False,
                            'limit':50})
            if isinstance(rule, Adjust_position):
                for r3 in rule.rules:
                    if isinstance(r3, Buy_stocks_chan) or isinstance(r3, Buy_stocks_var):
                        r3.update_params(context, {'buy_count': g.buy_count, 'pos_control': context.port_pos_control})

    def __str__(self):
        return '参数自动调整'
    

# adjust stocks ###########

class Sell_stocks_chan(Sell_stocks):
    def __init__(self, params):
        Sell_stocks.__init__(self, params)
        self.monitor_levels = params.get('monitor_levels', ['5d','1d','60m'])
        self.money_fund = params.get('money_fund', ['511880.XSHG'])
    def handle_data(self, context, data):
        # 日线级别卖点
        cti = None
        TA_Factor.global_var = self.g
        # self.g.monitor_short_cm.updateGaugeStockList(newStockList=context.portfolio.positions.keys(), levels=[self.monitor_levels[-1]]) # gauge 30m level status
        if context.now.hour < 11: # 10点之前
            cti = checkTAIndicator_OR({
                'TA_Indicators':[
                                (TaType.BOLL_MACD,'1d',233),
                                # (TaType.BOLL_MACD,'60m',233),
                                # (TaType.MACD,'60m',233),
                                (TaType.MACD,'1d',233),
                                # (TaType.BOLL,'1d',100), 
                                ],
                'isLong':False}) 
        elif context.now.hour >= 14: # after 14:00
            cti = checkTAIndicator_OR({
                'TA_Indicators':[
                                # (TaType.BOLL,'240m',100), 
                                (TaType.BOLL_MACD,'240m',233), # moved from morning check
                                # (TaType.BOLL_MACD,'60m',233),
                                # (TaType.MACD,'60m',233),
                                (TaType.MACD,'240m',233),
                                # (TaType.TRIX_STATUS, '240m', 100),
                                (TaType.KDJ_CROSS, '240m', 100)
                    ], 
                'isLong':False}) 
        else:
            cti = checkTAIndicator_OR({
                'TA_Indicators':[
                                # (TaType.BOLL_MACD,'60m',233),
                                # (TaType.MACD,'60m',233),
                                ], 
                'isLong':False}) 
        to_sell = cti.filter(context, data, context.portfolio.positions.keys())
        to_sell = [stock for stock in to_sell if stock not in self.money_fund] # money fund only adjusted by buy method
        to_sell_intraday = self.intradayShortFilter(context, data)
        # to_sell_intraday = []
        try:
            to_sell = [stock for stock in to_sell if data[stock].close < data[stock].high_limit] # 涨停不卖
        except:
            pass

        # ML check
        to_sell_biaoli = context.mlb.gauge_stocks(context.portfolio.positions.keys(), isLong=False)
        
        to_sell = list(set(to_sell+to_sell_biaoli+to_sell_intraday))
        if to_sell:
            self.log.info('准备卖出:\n' + join_list(["[%s]" % (show_stock(x)) for x in to_sell], ' ', 10))
            self.adjust(context, data, to_sell)
            # remove stocks from short gauge
            sold_stocks = [stock for stock in to_sell if stock not in context.portfolio.positions.keys()] # make sure stock sold successfully
            # self.g.monitor_short_cm.displayMonitorMatrix(to_sell)
            # self.recordTrade(to_sell) # record all selling candidate
            # self.g.monitor_short_cm.removeGaugeStockList(sold_stocks)
        self.g.intraday_long_stock = [stock for stock in self.g.intraday_long_stock if stock in context.portfolio.positions.keys()]

    def intradayShortFilter(self, context, data):
        cti_short_check = None
        if context.now.hour < 11:
            cti_short_check = checkTAIndicator_OR({
                'TA_Indicators':[
                                # (TaType.BOLL,'60m',40),
                                (TaType.BOLL_MACD,'60m',233),
                                (TaType.MACD,'60m',233),
                                (TaType.TRIX_STATUS, '60m', 100),
                                ], 
                'isLong':False})
        elif context.now.hour >= 14:
            cti_short_check = checkTAIndicator_OR({
                'TA_Indicators':[
                                (TaType.BOLL_MACD,'60m',233),
                                (TaType.MACD,'60m',233),
                                # (TaType.BOLL,'60m',40),
                                (TaType.TRIX_STATUS, '60m', 100),
                                ], 
                'isLong':False})            
        else:
            # pass
            cti_short_check = checkTAIndicator_OR({
                'TA_Indicators':[
                                (TaType.BOLL_MACD,'60m',233),
                                (TaType.MACD,'60m',233),
                                # (TaType.BOLL,'60m',40),
                                (TaType.TRIX_STATUS, '60m', 100),
                                ], 
                'isLong':False})
        intraday_short_check = [stock for stock in context.portfolio.positions.keys() if stock in self.g.intraday_long_stock]
        to_sell = cti_short_check.filter(context, data, intraday_short_check) if cti_short_check else []
        return to_sell

    def before_trading_start(self, context):
        context.mlb = ML_biaoli_check({'threthold':0.95, 'rq':True, 'model_path':'cnn_lstm_model_index.h5','extra_training':True})

    def adjust(self, context, data, sell_stocks):
        # 卖出在待卖股票列表中的股票
        # 对于因停牌等原因没有卖出的股票则继续持有
        for stock in context.portfolio.positions.keys():
            if stock in sell_stocks:
                position = context.portfolio.positions[stock]
                self.g.close_position(self, position, True, data)
    
    def __str__(self):
        return '股票调仓卖出规则：卖出在对应级别卖点'

class Buy_stocks_chan(Buy_stocks):
    def __init__(self, params):
        Buy_stocks.__init__(self, params)
        self.buy_count = params.get('buy_count', 3)
        self.monitor_levels = params.get('monitor_levels', ['5d','1d','60m'])
        self.pos_control = params.get('pos_control', 1.0)
        self.daily_list = []
        
    def update_params(self, context, params):
        self.buy_count = params.get('buy_count', 3)
        self.pos_control = params.get('pos_control', 1.0)
        
    def handle_data(self, context, data):
        if self.is_to_return:
            self.log_warn('无法执行买入!! self.is_to_return 未开启')
            return
        if len([stock for stock in context.portfolio.positions if stock not in g.money_fund])==self.buy_count:
            self.log.info("满仓等卖")
            return

        if context.now.hour <= 10:
            self.daily_list = self.g.monitor_buy_list
            
        if not self.daily_list:
            self.log.info("现时无选股")
            return

        to_buy = self.daily_list
        # self.g.monitor_long_cm.updateGaugeStockList(newStockList=self.daily_list, levels=[self.monitor_levels[-1]])
        # 技术分析用于不买在卖点
        not_to_buy = self.dailyShortFilter(context, data, to_buy)
        
        self.daily_list = [stock for stock in self.daily_list if stock not in not_to_buy]

        to_buy = [stock for stock in to_buy if stock not in not_to_buy] 
        to_buy = [stock for stock in to_buy if stock not in context.portfolio.positions.keys()] 
        to_buy = [stock for stock in self.g.monitor_buy_list if stock in to_buy]
        try:
            to_buy = [stock for stock in to_buy if data[stock].close > data[stock].low_limit] # 跌停不买
        except:
            pass
        
        to_buy = self.dailyLongFilter(context, data, to_buy)
        
        if to_buy:
            buy_msg = '日内待买股:\n' + join_list(["[%s]" % (show_stock(x)) for x in to_buy], ' ', 10)
            self.log.info(buy_msg)
            self.adjust(context, data, to_buy)
            bought_stocks = [stock for stock in context.portfolio.positions.keys() if stock in to_buy]
            #transfer long gauge to short gauge
            # self.g.monitor_short_cm.appendStockList(self.g.monitor_long_cm.getGaugeStockList(bought_stocks))
            # self.g.monitor_long_cm.displayMonitorMatrix(to_buy)
            # self.recordTrade(bought_stocks)
            self.send_port_info(context)
        elif context.now.hour >= 14:
            self.adjust(context, data, [])
            self.send_port_info(context)

        self.g.intraday_long_stock = [stock for stock in self.g.intraday_long_stock if stock in context.portfolio.positions.keys()] # keep track of bought stocks
        if context.now.hour >= 14:
            self.log.info('日内60m标准持仓:\n' + join_list(["[%s]" % (show_stock(x)) for x in self.g.intraday_long_stock], ' ', 10))
    
    def dailyLongFilter(self, context, data, to_buy):
        to_buy_list = []
        TA_Factor.global_var = self.g 
        
        cta = checkTAIndicator_OR({
        'TA_Indicators':[
                        (TaType.MACD,'60m',233),
                        (TaType.BOLL_MACD, '60m', 233),
                        ],
        'isLong':True})
        to_buy_intraday_list = cta.filter(context, data, to_buy)      

        # intraday short check
        intraday_not_to_buy = self.intradayShortFilter(context, data, to_buy_intraday_list+self.g.intraday_long_stock)
        to_buy_intraday_list = [stock for stock in to_buy_intraday_list if stock not in intraday_not_to_buy]
        
        # 日内待选股票中排除日内出卖点
        self.daily_list = [stock for stock in self.daily_list if stock not in intraday_not_to_buy] 
        # combine with existing intraday long stocks
        self.g.intraday_long_stock = list(set(self.g.intraday_long_stock + to_buy_intraday_list))

        if context.now.hour >= 14:
            cta = checkTAIndicator_OR({
            'TA_Indicators':[
                            (TaType.RSI, '240m', 100),
                            ],
            'isLong':True})
            to_buy_special_list = cta.filter(context, data, to_buy)
            # combine with existing intraday long stocks
            self.g.intraday_long_stock = list(set(self.g.intraday_long_stock + to_buy_special_list))            
            
            cta = checkTAIndicator_OR({
            'TA_Indicators':[
                            (TaType.MACD,'240m',233),
                            (TaType.BOLL_MACD, '240m', 233),
#                             (TaType.RSI, '240m', 100),
                            (TaType.KDJ_CROSS, '240m', 100),
                            ],
            'isLong':True})
            to_buy_list = cta.filter(context, data, to_buy)
            
            # ML check
            to_buy_list = context.mlb.gauge_stocks(to_buy_list, isLong=True)
            
            # 之前的日内选股的票排除掉如果被更大级别买点覆盖
            self.g.intraday_long_stock = [stock for stock in self.g.intraday_long_stock if stock not in to_buy_list]        
        return to_buy_list + to_buy_intraday_list

    def intradayShortFilter(self, context, data, to_buy):
        cti_short_check = None
        if context.now.hour < 11:
            cti_short_check = checkTAIndicator_OR({
                'TA_Indicators':[
                                (TaType.MACD,'60m',233),
                                (TaType.BOLL_MACD,'60m',233),
                                # (TaType.BOLL,'60m',40),
                                (TaType.TRIX_STATUS, '60m', 100)
                                ], 
                'isLong':False})
        elif context.now.hour >= 14:
            cti_short_check = checkTAIndicator_OR({
                'TA_Indicators':[
                                (TaType.MACD,'60m',233),
                                (TaType.BOLL_MACD,'60m',233),
                                # (TaType.BOLL,'60m',40),
                                (TaType.TRIX_STATUS, '60m', 100)
                                ], 
                'isLong':False})  
        else:
            cti_short_check = checkTAIndicator_OR({
                'TA_Indicators':[
                                (TaType.MACD,'60m',233),
                                (TaType.BOLL_MACD,'60m',233),
                                # (TaType.BOLL,'60m',40),
                                (TaType.TRIX_STATUS, '60m', 100)
                                ], 
                'isLong':False})  

        not_to_buy = cti_short_check.filter(context, data, to_buy) if cti_short_check else []
        return not_to_buy


    def dailyShortFilter(self, context, data, to_buy):
        remove_from_candidate = []
        not_to_buy = []
        cti_short_check = checkTAIndicator_OR({
            'TA_Indicators':[
                            # (TaType.MACD,'60m',233),
                            # (TaType.BOLL_MACD,'60m',233),
                            ], 
            'isLong':False})
        not_to_buy += cti_short_check.filter(context, data, to_buy)       
    
        if context.now.hour < 11:
            cti_short_check = checkTAIndicator_OR({
                'TA_Indicators':[
                                (TaType.MACD,'1d',233),
                                (TaType.BOLL_MACD,'1d',233),
                                (TaType.BOLL,'1d',40),
                                (TaType.TRIX_STATUS, '1d', 100)], 
                'isLong':False})
            remove_from_candidate = cti_short_check.filter(context, data, to_buy) 
            not_to_buy += remove_from_candidate
        
        elif context.now.hour >= 14:
            cti_short_check = checkTAIndicator_OR({
                'TA_Indicators':[
                                (TaType.MACD,'240m',233),
                                (TaType.BOLL_MACD,'240m',233),
                                (TaType.BOLL,'240m',40),
                                (TaType.TRIX_STATUS, '240m', 100), 
                                ], 
                'isLong':False})
            remove_from_candidate = cti_short_check.filter(context, data, to_buy)
            not_to_buy += remove_from_candidate

        # not_to_buy += self.g.monitor_long_cm.filterUpTrendDownTrend(stock_list=to_buy, level_list=self.monitor_levels[1:], update_df=False)
        # not_to_buy += self.g.monitor_long_cm.filterUpTrendUpNode(stock_list=to_buy, level_list=self.monitor_levels[1:], update_df=False)
        # not_to_buy += self.g.monitor_long_cm.filterUpNodeDownTrend(stock_list=to_buy, level_list=self.monitor_levels[1:], update_df=False)
        # not_to_buy += self.g.monitor_long_cm.filterUpNodeUpNode(stock_list=to_buy, level_list=self.monitor_levels[1:], update_df=False)
        
        ## not_to_buy += self.g.monitor_long_cm.filterDownNodeUpNode(stock_list=to_buy, level_list=self.monitor_levels[1:], update_df=False)
        not_to_buy = list(set(not_to_buy))
        # 大级别卖点从待选股票中去掉
#         if remove_from_candidate:
#             self.g.monitor_long_cm.removeGaugeStockList(remove_from_candidate)
#             self.g.monitor_buy_list = [stock for stock in self.g.monitor_buy_list if stock not in remove_from_candidate]
        return not_to_buy
    
    def adjust(self, context, data, buy_stocks):
        # 买入股票
        # 始终保持持仓数目为g.buy_stock_count
        # 根据股票数量分仓
        # 此处只根据可用金额平均分配购买，不能保证每个仓位平均分配
        position_count = len(context.portfolio.positions)
        if self.buy_count > position_count:
            value = context.portfolio.total_value * self.pos_control / self.buy_count
            for stock in buy_stocks:
                if stock in self.g.sell_stocks:
                    continue
                if context.portfolio.positions[stock].total_amount == 0:
                    if self.g.open_position(self, stock, value, 0):
                        if len(context.portfolio.positions) == self.buy_count:
                            break

    def after_trading_end(self, context):
        self.g.sell_stocks = []
        self.daily_list = []

    def __str__(self):
        return '股票调仓买入规则：买在对应级别买点'
        

class Buy_stocks_var(Buy_stocks_chan):
    """使用 VaR 方法做调仓控制"""
    def __init__(self, params):
        Buy_stocks_chan.__init__(self, params)
        self.money_fund = params.get('money_fund', ['511880.XSHG'])
        self.adjust_pos = params.get('adjust_pos', True)
        self.equal_pos = params.get('equal_pos', False)
        self.p_value = params.get('p_val', 2.58)
        self.risk_var = params.get('risk_var', 0.13)
        self.pc_var = None

    def adjust(self, context, data, buy_stocks):
        if not self.pc_var:
            # 设置 VaR 仓位控制参数。风险敞口: 0.05,
            # 正态分布概率表，标准差倍数以及置信率: 0.96, 95%; 2.06, 96%; 2.18, 97%; 2.34, 98%; 2.58, 99%; 5, 99.9999%
            # 赋闲资金可以买卖银华日利做现金管理: ['511880.XSHG']
            self.pc_var = PositionControlVar(context, self.risk_var, self.p_value, self.money_fund, self.equal_pos)
        if self.is_to_return:
            self.log_warn('无法执行买入!! self.is_to_return 未开启')
            return
        
        if self.adjust_pos:
            self.adjust_all_pos(context, data, buy_stocks)
        else:
            self.adjust_new_pos(context, data, buy_stocks)
    
    def adjust_new_pos(self, context, data, buy_stocks):
        position_count = len([stock for stock in context.portfolio.positions.keys() if stock not in self.money_fund and stock not in buy_stocks])
        trade_ratio = {}
        if self.buy_count > position_count:
            buy_num = self.buy_count - position_count
            trade_ratio = self.pc_var.buy_the_stocks(context, buy_stocks[:buy_num])
        else:
            trade_ratio = self.pc_var.func_rebalance(context)

        # sell money_fund if not in list
        for stock in context.portfolio.positions.keys():
            position = context.portfolio.positions[stock]
            if stock in self.money_fund: 
                if (stock not in trade_ratio or trade_ratio[stock] == 0.0):
                    self.g.close_position(self, position, True, data)
                else:
                    self.g.open_position(self, stock, context.portfolio.total_value*trade_ratio[stock],0)
                    
        for stock in trade_ratio:
            if stock in self.g.sell_stocks and stock not in self.money_fund:
                continue
            if context.portfolio.positions[stock].total_amount == 0:
                if self.g.open_position(self, stock, context.portfolio.total_value*trade_ratio[stock],0):
                    if len(context.portfolio.positions) == self.buy_count+1:
                        break        
        
    def adjust_all_pos(self, context, data, buy_stocks):
        # 买入股票或者进行调仓
        # 始终保持持仓数目为g.buy_count
        to_buy_num = len(buy_stocks)
        # exclude money_fund
        holding_positon_exclude_money_fund = [stock for stock in context.portfolio.positions.keys() if stock not in self.money_fund]
        position_count = len(holding_positon_exclude_money_fund)
        trade_ratio = {}
        if self.buy_count <= position_count+to_buy_num: # 满仓数
            buy_num = self.buy_count - position_count
            trade_ratio = self.pc_var.buy_the_stocks(context, holding_positon_exclude_money_fund+buy_stocks[:buy_num])
        else: # 分仓数
            trade_ratio = self.pc_var.buy_the_stocks(context, holding_positon_exclude_money_fund+buy_stocks)

        current_ratio = self.g.getCurrentPosRatio(context)
        order_stocks = self.getOrderByRatio(current_ratio, trade_ratio)
        for stock in order_stocks:
            if stock in self.g.sell_stocks:
                continue
            if self.g.open_position(self, stock, context.portfolio.total_value*trade_ratio[stock],0):
                pass
    
    def getOrderByRatio(self, current_ratio, target_ratio):
        diff_ratio = [(stock, target_ratio[stock]-current_ratio[stock]) for stock in target_ratio if stock in current_ratio] \
                    + [(stock, target_ratio[stock]) for stock in target_ratio if stock not in current_ratio] \
                    + [(stock, 0.0) for stock in current_ratio if stock not in target_ratio]
        diff_ratio.sort(key=lambda x: x[1]) # asc
        return [stock for stock,_ in diff_ratio]
    
    def __str__(self):
        return '股票调仓买入规则：使用 VaR 方式买入或者调整股票达目标股票数'

