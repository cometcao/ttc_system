# -*- encoding: utf8 -*-
'''
Created on 4 Dec 2017

@author: MetalInvest
'''
try:
    from rqdatac import *
except:
    pass
from securityDataManager import *
from oop_strategy_frame import *
from oop_select_stock import Filter_stock_list

# 排序基本类 共用指定参数为 weight
class SortBase(Rule):
    @property
    def weight(self):
        return self._params.get('weight', 1)

    @property
    def is_asc(self):
        return self._params.get('sort', SortType.asc) == SortType.asc

    def _sort_type_str(self):
        return '从小到大' if self.is_asc else '从大到小'

    def sort(self, context, data, stock_list):
        return stock_list


# '''--多因子计算：每个规则产生一个排名，并根据排名和权重进行因子计算--'''
class SortRules(Group_rules, Filter_stock_list):
    def filter(self, context, data, stock_list):
        self.log.info(join_list([show_stock(stock) for stock in stock_list[:10]], ' ', 10))
        sorted_stocks = []
        total_weight = 0  # 总权重。
        for rule in self.rules:
            if isinstance(rule, SortBase):
                total_weight += rule.weight
        for rule in self.rules:
            if not isinstance(rule, SortBase):
                continue
            if rule.weight == 0:
                continue  # 过滤权重为0的排序规则，为以后批量自动调整权重作意外准备
            stocks = stock_list[:]  # 为防排序规则搞乱list，每次都重新复制一份
            # 获取规则排序
            tmp_stocks = rule.sort(context, data, stocks)
            rule.log.info(join_list([show_stock(stock) for stock in tmp_stocks[:10]], ' ', 10))

            for stock in stock_list:
                # 如果被评分器删除，则不增加到总评分里
                if stock not in tmp_stocks:
                    stock_list.remove(stock)

            sd = {}
            rule_weight = rule.weight * 1.0 / total_weight
            for i, stock in enumerate(tmp_stocks):
                sd[stock] = (i + 1) * rule_weight
            sorted_stocks.append(sd)
        result = []

        for stock in stock_list:
            total_score = 0
            for sd in sorted_stocks:
                score = sd.get(stock, 0)
                if score == 0:  # 如果评分为0 则直接不再统计其它的
                    total_score = 0
                    break
                else:
                    total_score += score
            if total_score != 0:
                result.append([stock, total_score])
        result = sorted(result, key=lambda x: x[1])
        # 仅返回股票列表 。
        return [stock for stock, score in result]

    def __str__(self):
        return '多因子权重排序器'


# 按N日增长率排序
# day 指定按几日增长率计算,默认为20
class Sort_growth_rate(SortBase):
    def sort(self, context, data, stock_list):
        day = self._params.get('day', 20)
        r = []
        for stock in stock_list:
            rate = get_growth_rate(stock, day)
            if rate != 0:
                r.append([stock, rate])
        r = sorted(r, key=lambda x: x[1], reverse=not self.is_asc)
        return [stock for stock, rate in r]

    def __str__(self):
        return '[权重: %s ] [排序: %s ] 按 %d 日涨幅排序' % (self.weight, self._sort_type_str(), self._params.get('day', 20))


class Sort_price(SortBase):
    def sort(self, context, data, stock_list):
        r = []
        price_type = self._params.get('price_type', PriceType.now)
        if price_type == PriceType.now:
            for stock in stock_list:
                close = data[stock].close
                r.append([stock, close])
        elif price_type == PriceType.today_open:
            curr_data = get_current_data()
            for stock in stock_list:
                r.append([stock, curr_data[stock].day_open])
        elif price_type == PriceType.pre_day_open:
            stock_data = history(count=1, unit='1d', field='open', security_list=stock_list, df=False, skip_paused=True)
            for stock in stock_data:
                r.append([stock, stock_data[stock][0]])
        elif price_type == PriceType.pre_day_close:
            stock_data = history(count=1, unit='1d', field='close', security_list=stock_list, df=False,
                                 skip_paused=True)
            for stock in stock_data:
                r.append([stock, stock_data[stock][0]])
        elif price_type == PriceType.ma:
            n = self._params.get('period', 20)
            stock_data = history(count=n, unit='1d', field='close', security_list=stock_list, df=False,
                                 skip_paused=True)
            for stock in stock_data:
                r.append([stock, stock_data[stock].mean()])

        r = sorted(r, key=lambda x: x[1], reverse=not self.is_asc)
        return [stock for stock, close in r]

    def __str__(self):
        s = '[权重: %s ] [排序: %s ] 按当 %s 价格排序' % (
            self.weight, self._sort_type_str(), str(self._params.get('price_type', PriceType.now)))
        if self._params.get('price_type', PriceType.now) == PriceType.ma:
            s += ' [%d 日均价]' % (self._params.get('period', 20))
        return s


# --- 按换手率排序 ---
class Sort_turnover_ratio(SortBase):
    def sort(self, context, data, stock_list):
        q = query(valuation.code, valuation.turnover_ratio).filter(
            valuation.code.in_(stock_list)
        )
        if self.is_asc:
            q = q.order_by(valuation.turnover_ratio.asc())
        else:
            q = q.order_by(valuation.turnover_ratio.desc())
        stock_list = list(get_fundamentals(q)['code'])
        return stock_list

    def __str__(self):
        return '[权重: %s ] [排序: %s ] 按换手率排序 ' % (self.weight, self._sort_type_str())



# --- 按脉冲得分排序 ---
class Sort_cash_flow_rank(SortBase):
    def sort(self, context, data, stock_list):
        df = self.cow_stock_value(stock_list)
        return df.index
    
    def __str__(self):
        return '[权重: %s ] [排序: %s ] %s' % (self.weight, self._sort_type_str(), self.memo)
        
    def fall_money_day_3line(self, security_list,n, n1=20, n2=60, n3=160):
        def fall_money_count(money, n, n1, n2, n3):
            i = 0
            count = 0
            while i < n:
                money_MA200 = money[i:n3-1+i].mean()
                money_MA60 = money[i+n3-n2:n3-1+i].mean()
                money_MA20 = money[i+n3-n1:n3-1+i].mean()
                if money_MA20 <= money_MA60 and money_MA60 <= money_MA200 :
                    count = count + 1
                i = i + 1
            return count
    
        df = history(n+n3, unit='1d', field='money', security_list=security_list, skip_paused=True)
        s = df.apply(fall_money_count, args=(n,n1,n2,n3,))
        return s
    
    def money_5_cross_60(self, security_list,n, n1=5, n2=60):
        def money_5_cross_60_count(money, n, n1, n2):
            i = 0
            count = 0
            while i < n :
                money_MA60 = money[i+1:n2+i].mean()
                money_MA60_before = money[i:n2-1+i].mean()
                money_MA5 = money[i+1+n2-n1:n2+i].mean()
                money_MA5_before = money[i+n2-n1:n2-1+i].mean()
                if (money_MA60_before-money_MA5_before)*(money_MA60-money_MA5) < 0 : 
                    count=count+1
                i = i + 1
            return count
    
        df = history(n+n2+1, unit='1d', field='money', security_list=security_list, skip_paused=True)
        s = df.apply(money_5_cross_60_count, args=(n,n1,n2,))
        return s
    
    def cow_stock_value(self, security_list):
        sort_type = self._params.get('sort', SortType.desc)
        
        df = get_fundamentals(query(
                                    valuation.code, valuation.pb_ratio, valuation.circulating_market_cap
                                ).filter(
                                    valuation.code.in_(security_list),
                                    # valuation.circulating_market_cap <= 100
                                ))
        df.set_index('code', inplace=True, drop=True)
        s_fall = self.fall_money_day_3line(df.index.tolist(), 120, 20, 60, 160)
        s_cross = self.money_5_cross_60(df.index.tolist(), 120)
        df = pd.concat([df, s_fall, s_cross], axis=1, join='inner')
        df.columns = ['pb', 'cap', 'fall', 'cross']
        df['score'] = df['fall'] * df['cross'] / (df['pb']*(df['cap']**0.5))
        df.sort(['score'], ascending=(sort_type==SortType.asc), inplace=True) # ascending=False
        return(df)

# --- 按波动率排序 ---
class Sort_std_data(SortBase):
    def __init__(self, params):
        SortBase.__init__(self, params)
        self.period = params.get('period', 60)
    
    def sort(self, context, data, stock_list):
        dic_vol = {}
        for stock in stock_list:
            price = SecurityDataManager.get_data_rq(stock, count=60, period='1d', fields=['close'], skip_suspended=True, df=True, include_now=False)
            #vol = (price.pct_change().mean()/price.pct_change().std())['close']
            vol = price.pct_change().std()['close']
            #vol = (sqrt(price.pct_change().var())/price.pct_change().mean())['close']
            #vol = price.pct_change()['close'][-1]
            if vol >0:
                dic_vol[stock] = vol
        sort_vol = sorted(dic_vol.items(), key=(lambda d:d[1]),reverse=not self.is_asc)         #将波动率从大到小排列
        stockpool = [m for (m,n)  in sort_vol]
        return stockpool

    def __str__(self):
        return '[区间: %s ] [权重: %s ] [排序: %s ] %s' % (self.period, self.weight, self._sort_type_str(), self.memo)

# --- 按财务数据排序 ---
class Sort_financial_data(SortBase):
    def sort(self, context, data, stock_list):
        factor = eval(self._params.get('factor', None))
        if factor is None:
            return stock_list
        q = query(valuation, factor).filter(
            valuation.code.in_(stock_list)
        )
        if self.is_asc:
            q = q.order_by(factor.asc())
        else:
            q = q.order_by(factor.desc())
        stock_list = list(get_fundamentals(q)['code'])
        return stock_list

    def __str__(self):
        return '[权重: %s ] [排序: %s ] %s' % (self.weight, self._sort_type_str(), self.memo)


class Sort_rank_stock(SortBase):
    def sort(self, context, data, stock_list):
        rank_stock_count = self._params.get('rank_stock_count', None)
        if rank_stock_count:
            stock_list = stock_list[:rank_stock_count]
        dst_stocks = {}
        for stock in stock_list:
            h = SecurityDataManager.get_data_rq(stock, count=130, period='1d', fields=['close', 'high', 'low'], skip_suspended=True, df=True, include_now=False)
            low_price_130 = h.low.min()
            high_price_130 = h.high.max()
    
            avg_15 = data[stock].mavg(15, field='close')
            cur_price = data[stock].close
    
            #avg_15 = h['close'][-15:].mean()
            #cur_price = get_close_price(stock, 1, '1m')
    
            score = (cur_price-low_price_130) + (cur_price-high_price_130) + (cur_price-avg_15)
            #score = ((cur_price-low_price_130) + (cur_price-high_price_130) + (cur_price-avg_15)) / cur_price
            dst_stocks[stock] = score
            
        df = pd.DataFrame(dst_stocks.values(), index=dst_stocks.keys())
        df.columns = ['score']
        df = df.sort(columns='score', ascending=self.is_asc)
        return df.index

    def __str__(self):
        return '[权重: %s ] [排序: %s ] %s' % (self.weight, self._sort_type_str(), self.memo)


# '''------------------截取欲购股票数-----------------'''
class Filter_buy_count(Filter_stock_list):
    def __init__(self, params):
        self.buy_count = params.get('buy_count', 3)

    def update_params(self, context, params):
        self.buy_count = params.get('buy_count', self.buy_count)

    def filter(self, context, data, stock_list):
        if len(stock_list) > self.buy_count:
            return stock_list[:self.buy_count]
        else:
            return stock_list

    def __str__(self):
        return '获取最终待购买股票数:[ %d ]' % (self.buy_count)
