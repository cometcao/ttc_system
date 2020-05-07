# -*- encoding: utf-8 -*-

from utility.kBar_Chan import *
from utility.centralRegion import *
from utility.equilibrium import *

from pyecharts.charts import Kline,Scatter,Line,Grid,Bar
from pyecharts import options as opts

def draw_chan(stock, stock_df_fenduan, stock_df, kc, end_time, period): #
    stock_df_original = stock_df[['date', 'open', 'close', 'low', 'high']]
    stock_df_bi = kc.getFenBI_df()[['date','chan_price']]
    stock_df_xd = kc.getFenDuan_df()[['date','chan_price']]

#     overlap = Overlap(width=1500, height=600)
#     overlap.use_theme( "dark")

    kline = (
#         Kline({"theme": ThemeType.DARK})
        Kline()
        .add_xaxis(xaxis_data= stock_df_original['date'].tolist(), )
        .add_yaxis(
            series_name=stock,
            y_axis=stock_df_original[['open', 'close', 'low', 'high']].tolist(),
            itemstyle_opts=opts.ItemStyleOpts(color="#ec0000",color0="#00da3c"), #添加用 Hex 字符串表示的红和绿两种颜色，对应着 K线涨和跌的颜色。
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(
                title="缠论",
            ),
            xaxis_opts=opts.AxisOpts(type_="category"),
            # 分割线
            yaxis_opts=opts.AxisOpts(
                is_scale=True,
                splitarea_opts=opts.SplitAreaOpts(
                    is_show=True,
                    areastyle_opts=opts.AreaStyleOpts(opacity=1)
                ),
            ),
            legend_opts=opts.LegendOpts(is_show=True),
            toolbox_opts=opts.ToolboxOpts(),
            datazoom_opts=[
                opts.DataZoomOpts(
                    is_show=False,
                    type_="inside",
                    #xaxis_index=[0, 1],
                    range_start=0,  #添加两个「数据区域缩放」功能，一个看的到（用鼠标拉缩图最下面的 slider），一个看不到（用鼠标直接在图中拉缩），并且设置 xaxis_index =[0,1]，表示用 K 线图（index 0）来控制柱状图（index 1）。
                    range_end=100,
                ),
                opts.DataZoomOpts(
                    is_show=True,
                    #xaxis_index=[0, 1],
                    type_="slider",
                    pos_top="90%",
                    range_start=0,  #index为1和2的两幅图的数据局部伸缩跟着index0那幅图，这样就实现了用一根x轴的slider可以任意缩放三幅图的数据
                    range_end=100,                        
                ),
            ],
            tooltip_opts=opts.TooltipOpts(
                trigger="axis",
                axis_pointer_type="cross", #将两幅图的提示框合并在一起。
                background_color="rgba(245, 245, 245, 0.8)",
                border_width=1,
                border_color="#ccc",
                textstyle_opts=opts.TextStyleOpts(color="#000"),
            ),
            visualmap_opts=opts.VisualMapOpts(
                is_show=False,
                dimension=2,
                series_index=3,
                is_piecewise=True,
                pieces=[
                    {"value": -1, "color": "#ec0000"},
                    {"value": 1, "color": "#00da3c"},
                ],
            ), #坐标轴指示器配置和区域选择组件配置使得数据和轴可以一起联动。
            axispointer_opts=opts.AxisPointerOpts(
                is_show=True,
                link=[{"xAxisIndex": "all"}],
                label=opts.LabelOpts(background_color="#777"),
            ),
            brush_opts=opts.BrushOpts(
                x_axis_index="all",
                brush_link="all",
                out_of_brush={"colorAlpha": 0.1},
                brush_type="linex",
            ),
        )
    )

    #kline.render(f"F:\缠论_{stock}_info.html")

    Biline = (
        Line()
        .add_xaxis(xaxis_data=stock_df_bi['date'].tolist())
        .add_yaxis(
                series_name='笔',
                y_axis=stock_df_bi['chan_price'].tolist(),
                is_smooth=False,
                is_connect_nones=True,
                is_hover_animation=False,
                #linestyle_opts=opts.LineStyleOpts(color="red",width=2, type_="dashed"),
                linestyle_opts=opts.LineStyleOpts(color="red",width=2, opacity=0.5),
                label_opts=opts.LabelOpts(is_show=True),  
        ) 
#         .set_global_opts(xaxis_opts=opts.AxisOpts(type_="category"))
        )
    
    Bi = kline.overlap(Biline)
    
    crp = CentralRegionProcess(stock_df_fenduan, kc, isdebug=False)
    crp.define_central_region()
    stock_zs_x, stock_zs_y = crp.convert_to_graph_data()
        
    if stock_df_xd is not None:
        for i in range(0, len(stock_zs_x), 2):  
            XD_line = (
                Line()
                .add_xaxis(xaxis_data=stock_df_xd['date'].tolist())
                .add_yaxis(
                        series_name='段',
                        y_axis=stock_df_xd['chan_price'].tolist(),

                        is_smooth=False,
                        is_connect_nones=True,
                        is_hover_animation=False,
                        #linestyle_opts=opts.LineStyleOpts(color="red",width=2, type_="dashed"),
                        linestyle_opts=opts.LineStyleOpts(color="blue",width=2, opacity=0.5),
                        label_opts=opts.LabelOpts(is_show=False),  
                ) 
                .set_series_opts(
                            markarea_opts=opts.MarkAreaOpts(is_silent=True, data=[[{'xAxis': stock_zs_x[i], 'yAxis': stock_zs_y[i]}, {'xAxis':stock_zs_x[i+1], 'yAxis': stock_zs_y[i+1]} ]])
                        )
    #             .set_global_opts(xaxis_opts=opts.AxisOpts(type_="category"))
                )

#     MA_line = (
#         Line()
#         .add_xaxis(xaxis_data=stock_df_original['date'].tolist(),)
#         .add_yaxis(
#             series_name="MA5",
#             y_axis= talib.SMA(stock_df['close'],timeperiod = 5).tolist(),
#             is_smooth=True,
#             is_hover_animation=False,
#             linestyle_opts=opts.LineStyleOpts(width=1, opacity=0.5),
#             label_opts=opts.LabelOpts(is_show=False),
#         )
#         .add_yaxis(
#             series_name="MA10",
#             y_axis= talib.SMA(stock_df['close'],timeperiod = 10).tolist(),
#             is_smooth=True,
#             is_hover_animation=False,
#             linestyle_opts=opts.LineStyleOpts(width=1, opacity=0.5),
#             label_opts=opts.LabelOpts(is_show=False),
#         )
#         .add_yaxis(
#             series_name="MA13",
#             y_axis= talib.SMA(stock_df['close'],timeperiod = 13).tolist(),
#             is_smooth=True,
#             is_hover_animation=False,
#             linestyle_opts=opts.LineStyleOpts(width=1, opacity=0.5),
#             label_opts=opts.LabelOpts(is_show=False),
#         )
#         .add_yaxis(
#             series_name="MA34",
#             y_axis= talib.SMA(stock_df['close'],timeperiod = 34).tolist(),
#             is_smooth=True,
#             is_hover_animation=False,
#             linestyle_opts=opts.LineStyleOpts(width=1, opacity=0.5),
#             label_opts=opts.LabelOpts(is_show=False),
#         )
#         .set_global_opts(xaxis_opts=opts.AxisOpts(type_="category"))
#     ) 
    
    overlap = Bi.overlap(XD_line)
#     overlap_MA = overlap.overlap(MA_line)
    print("it's here")
    
    if len(stock_zs_x) != 0:
        
        for i in range(0, len(stock_zs_x), 2):
            zs1_line = (
                Line()
                .add_xaxis(xaxis_data=stock_zs_x[i:i+2],)
                .add_yaxis(
                    series_name="中枢",
                    y_axis=np.append(stock_zs_y[i],stock_zs_y[i]),
                    is_smooth=False,
                    is_hover_animation=False,
                    linestyle_opts=opts.LineStyleOpts(color="red",width=1.5, opacity=0.5),
                    label_opts=opts.LabelOpts(is_show=False),
                )
                .set_global_opts(xaxis_opts=opts.AxisOpts(type_="category"))
            ) 
    
            zs2_line = (
                Line()
                .add_xaxis(xaxis_data=stock_zs_x[i:i+2],)
                .add_yaxis(
                    series_name="中枢",
                    y_axis=np.append(stock_zs_y[i+1:i+2], stock_zs_y[i+1:i+2]),
                    is_smooth=False,
                    is_hover_animation=False,
                       linestyle_opts=opts.LineStyleOpts(color="red",width=1.5, opacity=0.5),
                    label_opts=opts.LabelOpts(is_show=False),
                )
                .set_global_opts(xaxis_opts=opts.AxisOpts(type_="category"))
            ) 
    
            zs3_line = (
                Line()
                .add_xaxis(xaxis_data=np.append(stock_zs_x[i:i+1],stock_zs_x[i:i+1]))
                .add_yaxis(
                    series_name="中枢",
                    y_axis=stock_zs_y[i:i+2],
                    is_smooth=False,
                    is_hover_animation=False,
                    linestyle_opts=opts.LineStyleOpts(color="red",width=1.5, opacity=0.5),
                    label_opts=opts.LabelOpts(is_show=False),
                )
                .set_global_opts(xaxis_opts=opts.AxisOpts(type_="category"))
            ) 
    
            zs4_line = (
                Line()
                .add_xaxis(xaxis_data=np.append(stock_zs_x[i+1:i+2],stock_zs_x[i+1:i+2]))
                .add_yaxis(
                    series_name="中枢",
                    y_axis=stock_zs_y[i:i+2],
                    is_smooth=False,
                    is_hover_animation=False,
                    linestyle_opts=opts.LineStyleOpts(color="red",width=1.5, opacity=0.5),
                    label_opts=opts.LabelOpts(is_show=False),
                )
                .set_global_opts(xaxis_opts=opts.AxisOpts(type_="category"))
            ) 
    
            zs1 =  zs1_line.overlap(zs2_line)
            zs2 =  zs3_line.overlap(zs4_line)
            zs  =  zs1.overlap(zs2)
            overlap.overlap(zs).render("diagram/{0}@{1}#{2}.html".format(stock[0:6], end_time[0:10], period))
    print("it's done")


stock = '000150.XSHE'
# end_time= '2019-10-21    14:30:00'
end_time=pd.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
period = '30m'
stock_df=JqDataRetriever.get_bars(stock, 
                                   count=4800, 
                                   end_dt=end_time, 
                                   unit=period,
                                   fields= ['date', 'open',  'high', 'low','close', 'money'],
                                   fq_ref_date = datetime.datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S"),
                                   df=False)
# period='1m'
# stock_df=JqDataRetriever.get_bars(stock, 
#                    start_dt='2017-11-13 10:20:00', 
#                    end_dt=end_time, 
#                    unit=period,
#                    fields= ['date', 'open',  'high', 'low','close', 'money'],
#                    fq_ref_date = datetime.datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S"),
#                    df=False)
kc = KBarChan(stock_df, isdebug=False)
stock_df_fenduan = kc.getFenDuan()
# stock_df_fenduan = kc.getFenDuan(TopBotType.top)

draw_chan(stock, stock_df_fenduan, stock_df, kc, end_time, period) #