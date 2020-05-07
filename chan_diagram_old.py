# -*- encoding: utf-8 -*-
from utility.kBarProcessor import *
from utility.centralRegion import *
from utility.equilibrium import *
from pyecharts import Line, Overlap, Kline
from pyecharts import configure
configure(global_theme='dark')

def draw_chan(stock, stock_df): #
    stock_df_original = stock_df[['open', 'close', 'low', 'high']]
    stock_df_bi = stock_df[(stock_df['tb']==TopBotType.top) | (stock_df['tb']==TopBotType.bot)][['chan_price']]
    stock_df_xd = stock_df[(stock_df['xd_tb']==TopBotType.top) | (stock_df['xd_tb']==TopBotType.bot)][['chan_price']]
    
    crp = CentralRegionProcess(stock_df, isdebug=False)
    crp.define_central_region()
    stock_zs_x, stock_zs_y = crp.convert_to_graph_data()
    
    overlap = Overlap(width=1500, height=600)
    overlap.use_theme( "dark")
 
    kline = Kline("����")
    kline.use_theme( "dark")
    kline.add(stock, 
              [str(d) for d in stock_df_original.index], 
              stock_df_original.values, 
              is_datazoom_show=True,
              datazoom_range=[80,100], 
              xaxis_interval=10,
              xaxis_rotate=30, 
              background_color='black')
    overlap.add(kline)

    line_1 = Line()
    line_1.use_theme( "dark")
    line_1.add("�ֱ�", 
               [str(d) for d in stock_df_bi.index], 
               stock_df_bi.values, 
               line_color = 'yellow', 
               is_datazoom_show=True,
               xaxis_interval=10,
               xaxis_rotate=30)
    overlap.add(line_1)
    
    if stock_df_xd is not None:
        line_2 = Line()
        line_2.use_theme( "dark")
        line_2.add("�ֶ�", 
                   [str(d) for d in stock_df_xd.index], 
                   stock_df_xd.values, 
                   line_color = 'blue',
                   mark_point_symbol="arrow", 
                   is_datazoom_show=True,
                   xaxis_interval=10,
                   xaxis_rotate=30)    
        overlap.add(line_2)
    
#     print(stock_zs_x)
#     print(stock_zs_y)
    
    for i in range(0, len(stock_zs_x), 2):
        line_a = Line()
        line_a.use_theme( "dark")
        line_a.add("����", 
                   [str(d) for d in stock_zs_x[i:i+2]], 
                   stock_zs_y[i:i+2], 
                   line_color = 'red',
                   line_width = 3,
                   is_step=True, 
                   is_label_show=True,
                   is_datazoom_show=True,
                   xaxis_interval=10,
                   xaxis_rotate=30)    
        overlap.add(line_a)
        
    overlap.render('stock_chan.html')   
    
from jqdata import * 
stock = '600195.XSHG'
stock_df=get_price(stock, count=2000, end_date='2021-01-05', frequency='5m',fields= ['open',  'high', 'low','close', 'volume', 'money'])
kb = KBarProcessor(stock_df, isdebug=False)
stock_df_all = kb.getIntegradedXD()

draw_chan(stock, stock_df_all) #
    