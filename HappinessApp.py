#------------------------------------------ Librerias------------------------------------------------#
# from faulthandler import disable
from requests import options
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as plt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import os
import json
import nltk
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import plotly.express as px
import plotly.figure_factory as ff
from cv2 import transpose
from matplotlib import container
# mapas interactivos.
import geopandas as gpd
from branca.colormap import LinearColormap
import streamlit.components.v1 as components
import plotly.graph_objs as go
import chart_studio.plotly as py
from plotly.offline import iplot, init_notebook_mode
from streamlit_option_menu import option_menu


#------------------------------------------Configuración pagina-----------------------------------#
st.set_page_config(page_title="Happiness-by Fara", layout="wide", page_icon="😀") # después establecer el título de página, su layout e icono 
#titulo principal
original_title = ('<h1 style="font-family:sans-serif ; color: black;font-size: 60px;">World Happiness</h1>')
st.markdown(original_title, unsafe_allow_html=True)

#----------------------------------------Dataset y preparación variables--------------------------#
df155 = pd.read_csv("2015.csv",sep="[,]",engine="python")
df165 = pd.read_csv("2016.csv",sep="[,]",engine="python")
df175 = pd.read_csv("2017.csv",sep="[,]",engine="python")
df185 = pd.read_csv("2018.csv",sep="[,]",engine="python")
df195 = pd.read_csv("2019.csv",sep="[,]",engine="python")

df15=df155.drop(["Unnamed: 0"], axis=1, inplace=True)
df16=df165.drop(["Unnamed: 0"], axis=1, inplace=True)
df17=df175.drop(["Unnamed: 0"], axis=1, inplace=True)
df18=df185.drop(["Unnamed: 0"], axis=1, inplace=True)
df19=df195.drop(["Unnamed: 0"], axis=1, inplace=True)

df15=df155
df16=df165
df17=df175
df18=df185
df19=df195


#ocultar errores
st.set_option('deprecation.showPyplotGlobalUse', False)  
#------------------------------------------Menú izquierdo-----------------------------------------#

with st.sidebar:
    selected=option_menu(
        menu_title="Menú reporte",
        options=["Principal", "Análisis por año", "Comparativo"],
    )

#****************Principal********************#
if selected == "Principal":
    col1, col2 = st.columns(2)
    with col1: 
        st.image ('happypic.jpg')
    with col2:
        texto = open('ContextoHS.md', 'r', encoding='utf-8')
        st.markdown(texto.read(), unsafe_allow_html=True)

    st.write("---")
    col1, col2, col3 = st.columns(3)
    if col2.button('CONTENIDO DE LAS COLUMNAS'):
        """
        ✔️GDP Economy: el PIB per cápita es una medida de la producción económica de un país que tiene en cuenta su número de habitantes.

        ✔️Family: el apoyo social son los amigos, familias y otras personas a los que acudir. 

        ✔️Health: La esperanza de vida sana es el número medio de años que un recién nacido puede esperar vivir con "plena salud", es decir, sin que le impidan las enfermedades o lesiones incapacitantes.

        ✔️Freedom: La libertad de elección describe la oportunidad y la autonomía de un individuo para realizar una acción seleccionada de entre al menos dos opciones disponibles, sin estar limitado por partes externas.

        ✔️Generosity: la cualidad de ser amable y generoso.

        ✔️Corruption: el Índice de Percepción de la Corrupción (IPC) es un índice publicado anualmente por Transparencia Internacional que clasifica a los países "por sus niveles percibidos de corrupción en el sector público, según lo determinado por evaluaciones de expertos y encuestas de opinión".
         
        """

    st.write("---")
    col1, col2, col3 = st.columns(3)
    if col2.button('-MARCO DEL ANALISIS-'):
        """
        ✔️Se seleccionaron los 147 países que se repetian a lo largo de todos los años (2015-2019).

        ✔️Se realizó limpieza y organización de las columnas.

        ✔️Había un solo dato vacío, el cual se completó con la media de la columna.
                 """
    st.write("---")
    col1, col2, col3 = st.columns(3)
    if col2.button('OBJETIVO DEL ANALISIS'):
        """
        🎯 Analizaremos los datos de 147 países entre los años 2015 a 2019 para obtener respuestas a:

        *¿Qué países y regiones tienen los mejores puntajes de felicidad? ¿Cuáles los peores?

        *¿Cómo contribuyen a la felicidad cada uno de los seis factores?

        *¿Cómo cambiaron los puntuajes a través de los años?
             
             """
    st.write("---")
    col1, col2, col3 = st.columns(3)
    if col2.button('Click aquí para más detalles sobre el origen de los indicadores'):       
        """
        Para determinar cuál es el país más feliz del mundo, los investigadores analizaron los datos exhaustivos de las encuestas de Gallup de 149 países durante los últimos tres años, controlando específicamente los resultados en seis categorías concretas: producto interior bruto per cápita, apoyo social, esperanza de vida saludable,
        libertad para tomar tus propias decisiones vitales, generosidad de la población en general y percepción de los niveles de corrupción interna y externa.

        Para poder comparar adecuadamente los datos de cada país, los investigadores crearon un país ficticio -bautizado como Distopía- lleno de "la gente menos feliz del mundo". 
        A continuación, establecieron Dystopia como el valor más bajo en cada una de las seis categorías y midieron las puntuaciones de los países del mundo real con respecto a este valor. Las seis variables se mezclaron para crear una única puntuación combinada para cada país.
        Si se suman todos estos factores, se obtiene la puntuación de felicidad.

        https://worldhappiness.report/

        https://www.kaggle.com/datasets/mathurinache/world-happiness-report

        https://www.gallup.com/home.aspx
        """

    st.write("---")

#****************Por año ********************#
#     
if selected == "Análisis por año":
    opt= st.sidebar.radio("Escoje el año:", options=("2015","2016","2017","2018","2019"))

#AÑO 2015
    if opt == "2015":
        st.write('Head Dataset') 
        st.dataframe(df15.sort_values(by="Happiness Score", ascending=False).head())
        st.write('Tail Dataset')
        st.dataframe(df15.sort_values(by="Happiness Score", ascending=False).tail())

        st.write("---")

        st.markdown("<h4 style='text-align: left; color: black;'>Correlaciones</h4>", unsafe_allow_html=True)
        col1, col2 = st.columns((2,2), gap='large')
        with col1:
            corr = df15.corr(method = 'spearman', numeric_only=True)
            mask = np.triu(np.ones_like(corr, dtype=bool)) 
            fig15, ax15 = plt.subplots(figsize=(5,4))
            sns.heatmap(corr.iloc[1:7,0:6], mask=mask[1:7,0:6], cmap="YlGnBu", vmax=1,center=0,
            square=True, linewidths=.4,annot = True)
            st.pyplot(fig15)
        with col2:
            st.write("")
                     
            '''
            En este mapa de correlaciones, podemos observar:

            ➡️ Los factores más correlacionados con el puntaje de la felicidad son: ECONOMIA(GDP), FAMILIA(family) y VIDA SANA (health).   


            ➡️ La ECONOMIA tiene una muy fuerte correlación con la vida sana.


            ➡️ A su vez, la GENEROSIDAD tiene muy baja correlación con la ECONOMIA y con la VIDA SANA
            '''   
        st.write("---")
      
        st.markdown("<h4 style='text-align: left; color: black;'>Ranking países más y menos felices</h4>", unsafe_allow_html=True)
        col1, col2 = st.columns((2,2), gap='large')
        with col1:
            top15=df15.sort_values(by="Happiness Score", ascending=False).head()
            topfelices15 =px.histogram(top15,x="Happiness Score", y="Country", color= "Happiness Score", width=800, title="Top 5 países más felices", opacity=0.75)
            st.plotly_chart(topfelices15)  
        with col2:
            notop15=df15.sort_values(by="Happiness Score", ascending=False).tail()
            topnofelices15 =px.histogram(notop15,x="Happiness Score", y="Country", color= "Happiness Score", width=800, title="Top 5 países menos felices", opacity=0.75)
            st.plotly_chart(topnofelices15)

        col1, col2 = st.columns((2,2), gap='large')
        with col1:
            st.write("")
            '''
            ⬆️Los primeros 4 países más felices pertenecen a Western Europe

            '''
        with col2:
            st.write("")
            '''
            ⬆️Todos ellos pertenecen a la Region Africa

            '''

        st.write("---")

        st.markdown("<h4 style='text-align: left; color: black;'>Mapa Happiness Score</h4>", unsafe_allow_html=True)
        dfmap15 = px.data.gapminder()
        figmap15 = px.choropleth(df15, locations="Country",locationmode="country names", color="Happiness Score", hover_name="Country")
        st.plotly_chart(figmap15)

        st.write("---")
        st.markdown("<h4 style='text-align: left; color: black;'>Happiness Score por Región</h4>", unsafe_allow_html=True)
               
        figregion15 = px.bar(        
        df15,
        x = "Region",
        y = "Happiness Score",
        color="Happiness Score")
        st.plotly_chart(figregion15)
        
    
        if st.button("Tabla promedio de Happiness Score por Region"):
            region_lists=list(df15['Region'].unique())
            region_happiness_avg=[]
            for each in region_lists:
                region=df15[df15['Region']==each]
                region_happiness_rate=sum(region["Happiness Score"])/len(region)
                region_happiness_avg.append(region_happiness_rate)
            data_happiness=pd.DataFrame({'region':region_lists,'region_happiness_avg':region_happiness_avg})
            new_index_happiness=(data_happiness['region_happiness_avg'].sort_values(ascending=False)).index.values
            sorted_data_happiness = data_happiness.reindex(new_index_happiness)
            st.dataframe(sorted_data_happiness)

        st.write("---")

        st.markdown("<h4 style='text-align: left; color: black;'>¿Cómo se comportan los factores en los países más y menos felices?</h4>", unsafe_allow_html=True)
        factorstop = df15.iloc[:,:]
        # Creating trace1
        trace1 = go.Scatter(x = df15['Country'],
                    y = df15['Economy (GDP per Capita)'],
                    mode = "lines+markers",
                    name = "Economy",
                    marker = dict(color = 'red'),
                    text= df15.Country)

        # Creating trace2
        trace2 = go.Scatter(x = df15['Country'],
                    y = df15['Health (Life Expectancy)'],
                    mode = "lines+markers",
                    name = "Health (Life Expectancy)",
                    marker = dict(color = 'blue'),
                    text= df15.Country)

        # Creating trace3
        trace3 = go.Scatter(x = df15['Country'],
                    y = df15['Family'],
                    mode = "lines+markers",
                    name = "Family",
                    marker = dict(color = 'grey'),
                    text= df15.Country)

        # Creating trace4
        trace4 = go.Scatter(x = df15['Country'],
                    y = df15["Freedom"],
                    mode = "lines+markers",
                    name = "Freedom",
                    marker = dict(color = 'black'),
                    text= df15.Country)

       
        data = [trace1, trace2, trace3, trace4]
        layout = dict(title = 'Comparación amplia factores más correlacionados 2015',
              xaxis= dict(title= 'Countries',ticklen= 4,zeroline= False),
              hovermode="x unified")
        figfactores15 = dict(data = data,layout = layout)
        st.plotly_chart(figfactores15)

        st.markdown("<h5 style='text-align: left; color: black;'>Comparativo factores top 5 más y menos felices</h5>", unsafe_allow_html=True)

        fig00, axes00 = plt.subplots(nrows=2, ncols=2,constrained_layout=True,figsize=(7,3))
        sns.barplot(x='Economy (GDP per Capita)',y='Country', data=top15, ax=axes00[0,0],palette="Blues_d")
        sns.barplot(x='Economy (GDP per Capita)' ,y='Country', data=notop15, ax=axes00[0,1],palette="YlGn")
        sns.barplot(x='Health (Life Expectancy)' ,y='Country', data=top15, ax=axes00[1,0],palette='OrRd')
        sns.barplot(x='Health (Life Expectancy)' ,y='Country', data=notop15, ax=axes00[1,1],palette='YlOrBr')
        sns.set(font_scale=0.6)
        st.pyplot(fig00)
        fig00.tight_layout()

        fig001, axes001 = plt.subplots(nrows=2, ncols=2,constrained_layout=True,figsize=(7,3))
        sns.barplot(x='Family',y='Country', data=top15, ax=axes001[0,0],palette="Blues_d")
        sns.barplot(x='Family' ,y='Country', data=notop15, ax=axes001[0,1],palette="YlGn")
        sns.barplot(x='Freedom' ,y='Country', data=top15, ax=axes001[1,0],palette='OrRd')
        sns.barplot(x='Freedom' ,y='Country', data=notop15, ax=axes001[1,1],palette='YlOrBr')
        sns.set(font_scale=0.6)
        st.pyplot(fig001)
        fig001.tight_layout()

#AÑO 2016
    if opt=="2016":
        st.write('Head Dataset')
        st.dataframe(df16.sort_values(by="Happiness Score", ascending=False).head())
        st.write('Tail Dataset')
        st.dataframe(df16.sort_values(by="Happiness Score", ascending=False).tail())
        col1, col2 = st.columns((2,2), gap='large')
        with col1:
            corr = df16.corr(method = 'spearman').sort_values(by = 'Happiness Score', axis = 0, ascending = False).sort_values(by = 'Happiness Score', axis = 1, ascending = False)
            mask = np.triu(np.ones_like(corr, dtype=np.bool)) 
            fig16, ax16 = plt.subplots(figsize=(5,4))
            cmap = sns.diverging_palette(220, 10, as_cmap=True)
            sns.heatmap(corr.iloc[1:7,0:6], mask=mask[1:7,0:6], cmap="YlGnBu", vmax=1, center=0,
            square=True, linewidths=.8, cbar_kws={"shrink": .6}, annot = True)
            st.pyplot(fig16)
        with col2:
            st.write("")
            '''
        
            En este mapa de correlaciones, podemos observar:
            
            ➡️ Los factores más correlacionados con el puntaje de la felicidad son: ECONOMIA(GDP), FAMILIA(family) y VIDA SANA (health).   


            ➡️ La ECONOMIA tiene una muy fuerte correlación con la vida sana.


            ➡️ A su vez, la GENEROSIDAD tiene muy baja correlación con la ECONOMIA y con la VIDA SANA
            '''   
        st.write("---")

        top16=  df16.sort_values(by="Happiness Score", ascending=False).head()    
        notop16=  df16.sort_values(by="Happiness Score", ascending=False).tail()    
        st.markdown("<h4 style='text-align: left; color: black;'>Ranking países más y menos felices</h4>", unsafe_allow_html=True)
        col1, col2 = st.columns((2,2), gap='large')
        with col1:
            topfelices16 =px.histogram(top16.sort_values(by="Happiness Score"),x="Happiness Score", y="Country", color= "Happiness Score", width=800, title="Top 5 países más felices")
            st.plotly_chart(topfelices16)  
        with col2:
            topnofelices16 =px.histogram(notop16,x="Happiness Score", y="Country", color= "Happiness Score", width=800, title="Top 5 países menos felices")
            st.plotly_chart(topnofelices16)

        col1, col2 = st.columns((2,2), gap='large')
        with col1:
            st.write("")
            '''
            ⬆️Los primeros 4 países más felices pertenecen a Western Europe

            '''
        with col2:
            st.write("")
            '''
            ⬆️Todos ellos pertenecen a la Region Africa

            '''

        st.write("---")

        st.markdown("<h4 style='text-align: left; color: black;'>Mapa Happiness Score</h4>", unsafe_allow_html=True)
        df = px.data.gapminder()
        figmap16 = px.choropleth(df16, locations="Country",locationmode="country names", color="Happiness Score", hover_name="Country")
        st.plotly_chart(figmap16)

        st.write("---")

        st.markdown("<h4 style='text-align: left; color: black;'>Happiness Score por Región</h4>", unsafe_allow_html=True)        
        figregion16 = px.bar(        
        df16,
        x = "Region",
        y = "Happiness Score",
        color="Happiness Score")
        st.plotly_chart(figregion16)
        
    
        if st.button("Tabla promedio de Happiness Score por Region"):
            region_lists=list(df16['Region'].unique())
            region_happiness_avg=[]
            for each in region_lists:
                region=df16[df16['Region']==each]
                region_happiness_rate=sum(region["Happiness Score"])/len(region)
                region_happiness_avg.append(region_happiness_rate)
            data_happiness=pd.DataFrame({'region':region_lists,'region_happiness_avg':region_happiness_avg})
            new_index_happiness=(data_happiness['region_happiness_avg'].sort_values(ascending=False)).index.values
            sorted_data_happiness = data_happiness.reindex(new_index_happiness)
            st.dataframe(sorted_data_happiness)

        st.write("---")

        st.markdown("<h4 style='text-align: left; color: black;'>¿Cómo se comportan los factores en los países más y menos felices?</h4>", unsafe_allow_html=True)
        factorstop = df16.iloc[:,:]
        # Creating trace1
        trace1 = go.Scatter(x = df16['Country'],
                    y = df16['Economy (GDP per Capita)'],
                    mode = "lines+markers",
                    name = "Economy",
                    marker = dict(color = 'red'),
                    text= df16.Country)

        # Creating trace2
        trace2 = go.Scatter(x = df16['Country'],
                    y = df16['Health (Life Expectancy)'],
                    mode = "lines+markers",
                    name = "Health (Life Expectancy)",
                    marker = dict(color = 'blue'),
                    text= df16.Country)

        # Creating trace3
        trace3 = go.Scatter(x = df16['Country'],
                    y = df16['Family'],
                    mode = "lines+markers",
                    name = "Family",
                    marker = dict(color = 'grey'),
                    text= df16.Country)

        # Creating trace4
        trace4 = go.Scatter(x = df16['Country'],
                    y = df16["Freedom"],
                    mode = "lines+markers",
                    name = "Freedom",
                    marker = dict(color = 'black'),
                    text= df16.Country)

       
        data = [trace1, trace2, trace3, trace4]
        layout = dict(title = 'Comparación factores más correlacionados 2016',
              xaxis= dict(title= 'Countries',ticklen= 4,zeroline= False),
              hovermode="x unified")
        figfactores16 = dict(data = data,layout = layout)
        st.plotly_chart(figfactores16)

        st.markdown("<h5 style='text-align: left; color: black;'>Comparativo factores top 5 más y menos felices</h5>", unsafe_allow_html=True)

        fig01, axes01 = plt.subplots(nrows=2, ncols=2,constrained_layout=True,figsize=(7,3))
        sns.barplot(x='Economy (GDP per Capita)',y='Country', data=top16, ax=axes01[0,0],palette="Blues_d")
        sns.barplot(x='Economy (GDP per Capita)' ,y='Country', data=notop16, ax=axes01[0,1],palette="YlGn")
        sns.barplot(x='Health (Life Expectancy)' ,y='Country', data=top16, ax=axes01[1,0],palette='OrRd')
        sns.barplot(x='Health (Life Expectancy)' ,y='Country', data=notop16, ax=axes01[1,1],palette='YlOrBr')
        sns.set(font_scale=0.6)
        st.pyplot(fig01)
        fig01.tight_layout()

        fig011, axes011 = plt.subplots(nrows=2, ncols=2,constrained_layout=True,figsize=(7,3))
        sns.barplot(x='Family',y='Country', data=top16, ax=axes011[0,0],palette="Blues_d")
        sns.barplot(x='Family' ,y='Country', data=notop16, ax=axes011[0,1],palette="YlGn")
        sns.barplot(x='Freedom' ,y='Country', data=top16, ax=axes011[1,0],palette='OrRd')
        sns.barplot(x='Freedom' ,y='Country', data=notop16, ax=axes011[1,1],palette='YlOrBr')
        sns.set(font_scale=0.6)
        st.pyplot(fig011)
        fig011.tight_layout()

#AÑO 2017
    if opt == "2017":
        st.write('Head Dataset') 
        st.dataframe(df17.sort_values(by="Happiness Score", ascending=False).head())
        st.write('Tail Dataset')
        st.dataframe(df17.sort_values(by="Happiness Score", ascending=False).tail())
        col1, col2 = st.columns((2,2), gap='large')
        with col1:
            corr = df17.corr(method = 'spearman').sort_values(by = 'Happiness Score', axis = 0, ascending = False).sort_values(by = 'Happiness Score', axis = 1, ascending = False)
            mask = np.triu(np.ones_like(corr, dtype=np.bool))
            fig17, ax17 = plt.subplots(figsize=(5,4))
            cmap = sns.diverging_palette(220, 10, as_cmap=True)
            sns.heatmap(corr.iloc[1:7,0:6], mask=mask[1:7,0:6], cmap="YlGnBu", vmax=1, center=0,
            square=True, linewidths=.8, cbar_kws={"shrink": .6}, annot = True)
            st.pyplot(fig17)
        with col2:
            st.write("")
                     
            '''
        
            En este mapa de correlaciones, podemos observar:
            
            ➡️ Los factores más correlacionados con el puntaje de la felicidad son: ECONOMIA(GDP), FAMILIA(family) y VIDA SANA (health).   


            ➡️ La ECONOMIA tiene una muy fuerte correlación con la vida sana.


            ➡️ A su vez, la GENEROSIDAD tiene muy baja correlación con la ECONOMIA y con la VIDA SANA
            '''   
        st.write("---")
      
        st.markdown("<h4 style='text-align: left; color: black;'>Ranking países más y menos felices</h4>", unsafe_allow_html=True)
        col1, col2 = st.columns((2,2), gap='large')
        top17=  df17.sort_values(by="Happiness Score", ascending=False).head()    
        notop17=  df17.sort_values(by="Happiness Score", ascending=False).tail()
        with col1:
            topfelices17 =px.histogram(top17,x="Happiness Score", y="Country", color= "Happiness Score", width=800, title="Top 5 países más felices", opacity=0.75)
            st.plotly_chart(topfelices17)  
        with col2:
            topnofelices17 =px.histogram(notop17,x="Happiness Score", y="Country", color= "Happiness Score", width=800, title="Top 5 países menos felices", opacity=0.75)
            st.plotly_chart(topnofelices17)

        col1, col2 = st.columns((2,2), gap='large')
        with col1:
            st.write("")
            '''
            ⬆️Los primeros 4 países más felices pertenecen a Western Europe

            '''
        with col2:
            st.write("")
            '''
            ⬆️Todos ellos pertenecen a la Region Africa

            '''

        st.write("---")

        st.markdown("<h4 style='text-align: left; color: black;'>Mapa Happiness Score</h4>", unsafe_allow_html=True)
        dfmap17 = px.data.gapminder()
        figmap17 = px.choropleth(df17, locations="Country",locationmode="country names", color="Happiness Score", hover_name="Country")
        st.plotly_chart(figmap17)

        st.write("---")
        st.markdown("<h4 style='text-align: left; color: black;'>Happiness Score por Región</h4>", unsafe_allow_html=True)
               
        figregion17 = px.bar(        
        df17,
        x = "Region",
        y = "Happiness Score",
        color="Happiness Score")
        st.plotly_chart(figregion17)
        
    
        if st.button("Tabla promedio de Happiness Score por Region"):
            region_lists=list(df17['Region'].unique())
            region_happiness_avg=[]
            for each in region_lists:
                region=df17[df17['Region']==each]
                region_happiness_rate=sum(region["Happiness Score"])/len(region)
                region_happiness_avg.append(region_happiness_rate)
            data_happiness=pd.DataFrame({'region':region_lists,'region_happiness_avg':region_happiness_avg})
            new_index_happiness=(data_happiness['region_happiness_avg'].sort_values(ascending=False)).index.values
            sorted_data_happiness = data_happiness.reindex(new_index_happiness)
            st.dataframe(sorted_data_happiness)

        st.write("---")

        st.markdown("<h4 style='text-align: left; color: black;'>¿Cómo se comportan los factores en los países más y menos felices?</h4>", unsafe_allow_html=True)
        factorstop17 = df17.iloc[:,:]
        # Creating trace1
        trace1 = go.Scatter(x = df17['Country'],
                    y = df17['Economy (GDP per Capita)'],
                    mode = "lines+markers",
                    name = "Economy",
                    marker = dict(color = 'red'),
                    text= df17.Country)

        # Creating trace2
        trace2 = go.Scatter(x = df17['Country'],
                    y = df17['Health (Life Expectancy)'],
                    mode = "lines+markers",
                    name = "Health (Life Expectancy)",
                    marker = dict(color = 'blue'),
                    text= df17.Country)

        # Creating trace3
        trace3 = go.Scatter(x = df17['Country'],
                    y = df17['Family'],
                    mode = "lines+markers",
                    name = "Family",
                    marker = dict(color = 'grey'),
                    text= df17.Country)

        # Creating trace4
        trace4 = go.Scatter(x = df17['Country'],
                    y = df17["Freedom"],
                    mode = "lines+markers",
                    name = "Freedom",
                    marker = dict(color = 'black'),
                    text= df17.Country)

       
        data = [trace1, trace2, trace3, trace4]
        layout = dict(title = 'Comparación amplia factores más correlacionados 2017',
              xaxis= dict(title= 'Countries',ticklen= 4,zeroline= False),
              hovermode="x unified")
        figfactores17 = dict(data = data,layout = layout)
        st.plotly_chart(figfactores17)

        st.markdown("<h5 style='text-align: left; color: black;'>Comparativo factores top 5 más y menos felices</h5>", unsafe_allow_html=True)

        fig02, axes02 = plt.subplots(nrows=2, ncols=2,constrained_layout=True,figsize=(7,3))
        sns.barplot(x='Economy (GDP per Capita)',y='Country', data=top17, ax=axes02[0,0],palette="Blues_d")
        sns.barplot(x='Economy (GDP per Capita)' ,y='Country', data=notop17, ax=axes02[0,1],palette="YlGn")
        sns.barplot(x='Health (Life Expectancy)' ,y='Country', data=top17, ax=axes02[1,0],palette='OrRd')
        sns.barplot(x='Health (Life Expectancy)' ,y='Country', data=notop17, ax=axes02[1,1],palette='YlOrBr')
        sns.set(font_scale=0.6)
        st.pyplot(fig02)
        fig02.tight_layout()

        fig002, axes002 = plt.subplots(nrows=2, ncols=2,constrained_layout=True,figsize=(7,3))
        sns.barplot(x='Family',y='Country', data=top17, ax=axes002[0,0],palette="Blues_d")
        sns.barplot(x='Family' ,y='Country', data=notop17, ax=axes002[0,1],palette="YlGn")
        sns.barplot(x='Freedom' ,y='Country', data=top17, ax=axes002[1,0],palette='OrRd')
        sns.barplot(x='Freedom' ,y='Country', data=notop17, ax=axes002[1,1],palette='YlOrBr')
        sns.set(font_scale=0.6)
        st.pyplot(fig002)
        fig002.tight_layout()
        


#AÑO 2018
    if opt == "2018": 
        st.write('Head')
        st.dataframe(df18.sort_values(by="Happiness Score", ascending=False).head())
        st.write('Tail')
        st.dataframe(df18.sort_values(by="Happiness Score", ascending=False).tail())
        col1, col2 = st.columns((2,2), gap='large')
        with col1:
            corr = df18.corr(method = 'spearman').sort_values(by = 'Happiness Score', axis = 0, ascending = False).sort_values(by = 'Happiness Score', axis = 1, ascending = False)
            mask = np.triu(np.ones_like(corr, dtype=np.bool)) 
            fig18, ax18 = plt.subplots(figsize=(5,4))
            cmap = sns.diverging_palette(220, 10, as_cmap=True)
            sns.heatmap(corr.iloc[1:7,0:6], mask=mask[1:7,0:6], cmap="YlGnBu", vmax=1, center=0,
            square=True, linewidths=.8, cbar_kws={"shrink": .6}, annot = True)
            st.pyplot(fig18)
        with col2:
            st.write("")
                     
            '''
        
             En este mapa de correlaciones, podemos observar:
            
            ➡️ Los factores más correlacionados con el puntaje de la felicidad son: ECONOMIA(GDP), FAMILIA(family) y VIDA SANA (health).   


            ➡️ La ECONOMIA tiene una muy fuerte correlación con la vida sana.


            ➡️ A su vez, la GENEROSIDAD tiene muy baja correlación con la ECONOMIA y con la VIDA SANA
            '''   
        st.write("---")
      
        st.markdown("<h4 style='text-align: left; color: black;'>Ranking países más y menos felices</h4>", unsafe_allow_html=True)
        col1, col2 = st.columns((2,2), gap='large')
        top18=  df18.sort_values(by="Happiness Score", ascending=False).head()    
        notop18=  df18.sort_values(by="Happiness Score", ascending=False).tail()
        with col1:
            topfelices18 =px.histogram(top18,x="Happiness Score", y="Country", color= "Happiness Score", width=800, title="Top 5 países más felices", opacity=0.75)
            st.plotly_chart(topfelices18)  
        with col2:
            topnofelices18 =px.histogram(notop18,x="Happiness Score", y="Country", color= "Happiness Score", width=800, title="Top 5 países menos felices", opacity=0.75)
            st.plotly_chart(topnofelices18)

        col1, col2 = st.columns((2,2), gap='large')
        with col1:
            st.write("")
            '''
            ⬆️Los primeros 4 países más felices pertenecen a Western Europe

            '''
        with col2:
            st.write("")
            '''
            ⬆️Todos ellos pertenecen a la Region Africa

            '''

        st.write("---")

        st.markdown("<h4 style='text-align: left; color: black;'>Mapa Happiness Score</h4>", unsafe_allow_html=True)
        dfmap18 = px.data.gapminder()
        figmap18 = px.choropleth(df18, locations="Country",locationmode="country names", color="Happiness Score", hover_name="Country")
        st.plotly_chart(figmap18)

        st.write("---")
        st.markdown("<h4 style='text-align: left; color: black;'>Happiness Score por Región</h4>", unsafe_allow_html=True)
               
        figregion18 = px.bar(        
        df18,
        x = "Region",
        y = "Happiness Score",
        color="Happiness Score")
        st.plotly_chart(figregion18)
        
    
        if st.button("Tabla promedio de Happiness Score por Region"):
            region_lists=list(df18['Region'].unique())
            region_happiness_avg=[]
            for each in region_lists:
                region=df18[df18['Region']==each]
                region_happiness_rate=sum(region["Happiness Score"])/len(region)
                region_happiness_avg.append(region_happiness_rate)
            data_happiness=pd.DataFrame({'region':region_lists,'region_happiness_avg':region_happiness_avg})
            new_index_happiness=(data_happiness['region_happiness_avg'].sort_values(ascending=False)).index.values
            sorted_data_happiness = data_happiness.reindex(new_index_happiness)
            st.dataframe(sorted_data_happiness)

        st.write("---")

        st.markdown("<h4 style='text-align: left; color: black;'>¿Cómo se comportan los factores en los países más y menos felices?</h4>", unsafe_allow_html=True)
        factorstop = df18.iloc[:,:]
        # Creating trace1
        trace1 = go.Scatter(x = df18['Country'],
                    y = df18['Economy (GDP per Capita)'],
                    mode = "lines+markers",
                    name = "Economy",
                    marker = dict(color = 'red'),
                    text= df18.Country)

        # Creating trace2
        trace2 = go.Scatter(x = df18['Country'],
                    y = df18['Health (Life Expectancy)'],
                    mode = "lines+markers",
                    name = "Health (Life Expectancy)",
                    marker = dict(color = 'blue'),
                    text= df18.Country)

        # Creating trace3
        trace3 = go.Scatter(x = df18['Country'],
                    y = df18['Family'],
                    mode = "lines+markers",
                    name = "Family",
                    marker = dict(color = 'grey'),
                    text= df18.Country)

        # Creating trace4
        trace4 = go.Scatter(x = df18['Country'],
                    y = df18["Freedom"],
                    mode = "lines+markers",
                    name = "Freedom",
                    marker = dict(color = 'black'),
                    text= df18.Country)

       
        data = [trace1, trace2, trace3, trace4]
        layout = dict(title = 'Comparación amplia factores más correlacionados 2018',
              xaxis= dict(title= 'Countries',ticklen= 4,zeroline= False),
              hovermode="x unified")
        figfactores18 = dict(data = data,layout = layout)
        st.plotly_chart(figfactores18)

        st.markdown("<h5 style='text-align: left; color: black;'>Comparativo factores top 5 más y menos felices</h5>", unsafe_allow_html=True)
  
        fig03, axes03 = plt.subplots(nrows=2, ncols=2,constrained_layout=True,figsize=(7,3))
        sns.barplot(x='Economy (GDP per Capita)',y='Country', data=top18, ax=axes03[0,0],palette="Blues_d")
        sns.barplot(x='Economy (GDP per Capita)' ,y='Country', data=notop18, ax=axes03[0,1],palette="YlGn")
        sns.barplot(x='Health (Life Expectancy)' ,y='Country', data=top18, ax=axes03[1,0],palette='OrRd')
        sns.barplot(x='Health (Life Expectancy)' ,y='Country', data=notop18, ax=axes03[1,1],palette='YlOrBr')
        sns.set(font_scale=0.6)
        st.pyplot(fig03)
        fig03.tight_layout()

        fig003, axes003 = plt.subplots(nrows=2, ncols=2,constrained_layout=True,figsize=(7,3))
        sns.barplot(x='Family',y='Country', data=top18, ax=axes003[0,0],palette="Blues_d")
        sns.barplot(x='Family' ,y='Country', data=notop18, ax=axes003[0,1],palette="YlGn")
        sns.barplot(x='Freedom' ,y='Country', data=top18, ax=axes003[1,0],palette='OrRd')
        sns.barplot(x='Freedom' ,y='Country', data=notop18, ax=axes003[1,1],palette='YlOrBr')
        sns.set(font_scale=0.6)
        st.pyplot(fig003)
        fig003.tight_layout()



#AÑO 2019
    if opt == "2019":
        st.write('Head Dataset') 
        st.dataframe(df19.sort_values(by="Happiness Score", ascending=False).head())
        st.write('Tail Dataset')
        st.dataframe(df19.sort_values(by="Happiness Score", ascending=False).tail())
        col1, col2 = st.columns((2,2), gap='large')
        with col1:
            corr = df19.corr(method = 'spearman').sort_values(by = 'Happiness Score', axis = 0, ascending = False).sort_values(by = 'Happiness Score', axis = 1, ascending = False)
            mask = np.triu(np.ones_like(corr, dtype=np.bool)) 
            fig19, ax19 = plt.subplots(figsize=(5,4))
            cmap = sns.diverging_palette(220, 10, as_cmap=True)
            sns.heatmap(corr.iloc[1:7,0:6], mask=mask[1:7,0:6], cmap="YlGnBu", vmax=1, center=0,
            square=True, linewidths=.8, cbar_kws={"shrink": .6}, annot = True)
            st.pyplot(fig19)
        with col2:
            st.write("")
                     
            '''
        
            En este mapa de correlaciones, podemos observar:
            
            ➡️ Los factores más correlacionados con el puntaje de la felicidad son: FAMILIA(family) que suma importancia este año,
            poniendose a la par de ECONOMIA(GDP) y por último VIDA SANA (health).   


            ➡️ La ECONOMIA tiene una muy fuerte correlación con la vida sana.


            ➡️ A su vez, la GENEROSIDAD tiene muy baja correlación con la ECONOMIA y con la VIDA SANA
            '''   
        st.write("---")
      
        st.markdown("<h4 style='text-align: left; color: black;'>Ranking países más y menos felices</h4>", unsafe_allow_html=True)
        col1, col2 = st.columns((2,2), gap='large')
        with col1:
            top19=df19.sort_values(by="Happiness Score", ascending=False).head()
            topfelices19 =px.histogram(top19,x="Happiness Score", y="Country", color= "Happiness Score", width=800, title="Top 5 países más felices", opacity=0.75)
            st.plotly_chart(topfelices19)  
        with col2:
            notop19=df19.sort_values(by="Happiness Score", ascending=False).tail()
            topnofelices19 =px.histogram(notop19,x="Happiness Score", y="Country", color= "Happiness Score", width=800, title="Top 5 países menos felices", opacity=0.75)
            st.plotly_chart(topnofelices19)

        col1, col2 = st.columns((2,2), gap='large')
        with col1:
            st.write("")
            '''
            ⬆️Todos los países más felices pertenecen a Western Europe

            '''
        with col2:
            st.write("")
            '''
            ⬆️4 de los 5 países menos felices pertenecen a la Region Africa

            '''

        st.write("---")

        st.markdown("<h4 style='text-align: left; color: black;'>Mapa Happiness Score</h4>", unsafe_allow_html=True)
        dfmap19 = px.data.gapminder()
        figmap19 = px.choropleth(df19, locations="Country",locationmode="country names", color="Happiness Score", hover_name="Country")
        st.plotly_chart(figmap19)

        st.write("---")
        st.markdown("<h4 style='text-align: left; color: black;'>Happiness Score por Región</h4>", unsafe_allow_html=True)
               
        figregion19 = px.bar(        
        df19,
        x = "Region",
        y = "Happiness Score",
        color="Happiness Score")
        st.plotly_chart(figregion19)
        
    
        if st.button("Tabla promedio de Happiness Score por Region"):
            region_lists=list(df19['Region'].unique())
            region_happiness_avg=[]
            for each in region_lists:
                region=df19[df19['Region']==each]
                region_happiness_rate=sum(region["Happiness Score"])/len(region)
                region_happiness_avg.append(region_happiness_rate)
            data_happiness=pd.DataFrame({'region':region_lists,'region_happiness_avg':region_happiness_avg})
            new_index_happiness=(data_happiness['region_happiness_avg'].sort_values(ascending=False)).index.values
            sorted_data_happiness = data_happiness.reindex(new_index_happiness)
            st.dataframe(sorted_data_happiness)

        st.write("---")

        st.markdown("<h4 style='text-align: left; color: black;'>¿Cómo se comportan los factores en los países más y menos felices?</h4>", unsafe_allow_html=True)
        factorstop = df19.iloc[:,:]
        # Creating trace1
        trace1 = go.Scatter(x = df19['Country'],
                    y = df19['Economy (GDP per Capita)'],
                    mode = "lines+markers",
                    name = "Economy",
                    marker = dict(color = 'red'),
                    text= df19.Country)

        # Creating trace2
        trace2 = go.Scatter(x = df19['Country'],
                    y = df19['Health (Life Expectancy)'],
                    mode = "lines+markers",
                    name = "Health (Life Expectancy)",
                    marker = dict(color = 'blue'),
                    text= df19.Country)

        # Creating trace3
        trace3 = go.Scatter(x = df19['Country'],
                    y = df19['Family'],
                    mode = "lines+markers",
                    name = "Family",
                    marker = dict(color = 'grey'),
                    text= df19.Country)

        # Creating trace4
        trace4 = go.Scatter(x = df19['Country'],
                    y = df19["Freedom"],
                    mode = "lines+markers",
                    name = "Freedom",
                    marker = dict(color = 'black'),
                    text= df19.Country)

       
        data = [trace1, trace2, trace3, trace4]
        layout = dict(title = 'Comparación amplia factores más correlacionados 2019',
              xaxis= dict(title= 'Countries',ticklen= 4,zeroline= False),
              hovermode="x unified")
        figfactores19 = dict(data = data,layout = layout)
        st.plotly_chart(figfactores19)

        st.markdown("<h5 style='text-align: left; color: black;'>Comparativo factores top 5 más y menos felices</h5>", unsafe_allow_html=True)
  
        fig04, axes04 = plt.subplots(nrows=2, ncols=2,constrained_layout=True,figsize=(7,3))
        sns.barplot(x='Economy (GDP per Capita)',y='Country', data=top19, ax=axes04[0,0],palette="Blues_d")
        sns.barplot(x='Economy (GDP per Capita)' ,y='Country', data=notop19, ax=axes04[0,1],palette="YlGn")
        sns.barplot(x='Health (Life Expectancy)' ,y='Country', data=top19, ax=axes04[1,0],palette='OrRd')
        sns.barplot(x='Health (Life Expectancy)' ,y='Country', data=notop19, ax=axes04[1,1],palette='YlOrBr')
        sns.set(font_scale=0.6)
        st.pyplot(fig04)
        fig04.tight_layout()

        fig004, axes004 = plt.subplots(nrows=2, ncols=2,constrained_layout=True,figsize=(7,3))
        sns.barplot(x='Family',y='Country', data=top19, ax=axes004[0,0],palette="Blues_d")
        sns.barplot(x='Family' ,y='Country', data=notop19, ax=axes004[0,1],palette="YlGn")
        sns.barplot(x='Freedom' ,y='Country', data=top19, ax=axes004[1,0],palette='OrRd')
        sns.barplot(x='Freedom' ,y='Country', data=notop19, ax=axes004[1,1],palette='YlOrBr')
        sns.set(font_scale=0.6)
        st.pyplot(fig004)
        fig004.tight_layout()



#****************Comparativo********************#

if selected == "Comparativo":
    st.title("Análisis comparativo 2015 y 2019")
    
    
    #DF con los 5 paises mas felices
    top_10_2015 = df15.sort_values(by="Happiness Score", ascending=False).head(10)
    top_10_2016 = df16.sort_values(by="Happiness Score", ascending=False).head(10)
    top_10_2017 = df17.sort_values(by="Happiness Score", ascending=False).head(10)
    top_10_2018 = df18.sort_values(by="Happiness Score", ascending=False).head(10)
    top_10_2019 = df19.sort_values(by="Happiness Score", ascending=False).head(10)

    # creating trace1
    uno =go.Scatter(
                    y = top_10_2015['Country'],
                    x = top_10_2015['Happiness Score'],
                    mode = "markers",
                    name = "2015",
                    marker = dict(size= 10,color = 'red'),
                    text= top_10_2015.Country)
    # # creating trace2
    # dos =go.Scatter(
    #                 y = top_10_2015['Country'],
    #                 x = top_10_2016['Happiness Score'],
    #                 mode = "markers",
    #                 name = "2016",
    #                 marker = dict(size= 10,color = 'green'),
    #                 text= top_10_2015.Country)
    # # creating trace3
    # tres =go.Scatter(
    #                 y = top_10_2015['Country'],
    #                 x = top_10_2017['Happiness Score'],
    #                 mode = "markers",
    #                 name = "2017",
    #                 marker = dict(size= 10,color = 'blue'),
    #                 text= top_10_2015.Country)
    # # creating trace4
    # cuatro =go.Scatter(
    #                 y = top_10_2015['Country'],
    #                 x = top_10_2018['Happiness Score'],
    #                 mode = "markers",
    #                 name = "2018",
    #                 marker = dict(size= 10,color = 'black'),
    #                 text= top_10_2015.Country)
    # creating trace5
    cinco =go.Scatter(
                    y = top_10_2019['Country'],
                    x = top_10_2019['Happiness Score'],
                    mode = "markers",
                    name = "2019",
                    marker = dict(size= 10,color = 'green'),
                    text= top_10_2019.Country)
    data = [uno, cinco]
    layout = dict(title = "Evolución 2015 vs 2019 top 10 más felices",
              xaxis= dict(title= 'Country',ticklen= 2,zeroline= False),
              yaxis= dict(title= 'Happiness',ticklen= 2,zeroline= False),
              hovermode="x unified"
             )
    fig = dict(data = data, layout = layout)
    st.plotly_chart(fig)
    
    st.write("")
    '''

    👉 En el 2015, el país más feliz fue Suiza con un puntaje de 7.59 mientras que en el 2019 fue Finlandia con un puntaje de 7.76.
    En el primer caso, se dio un retroseso de puntaje ya que pasó de 7.59 a 7.48 en 2019. En el segundo caso obervamos una mejora ya que pasó de 
    7.41 a 7.76

    👉 Observando el comportamiento de los factores explicativos se observa que ambos países tuvieron mejoras pero
    Finlandia tuvo un aumento mas importante. Por ejemplo:

    Economia 19 vs 15: Suiza aumentó un 3.60% | Finlandia 3.90%

    Familia 19 vs 15:  Suiza aumentó un 12.60% | Finlandia 19.70%

    Vida sana 19 vs 15: Suiza aumentó un 3% | Finlandia 10%

    👉A partir de aquí, podemos estimar que Suiza no "empeoró" como país sino que los demás países de caracteristicas similares
    obtuvieron mejoras mas importantes.


    '''
    st.write("---")
    #DF con los 10 paises mas felices

    notop_10_2015 = df15.sort_values(by="Happiness Score", ascending=False).tail(10)
    notop_10_2016 = df16.sort_values(by="Happiness Score", ascending=False).tail(10)
    notop_10_2017 = df17.sort_values(by="Happiness Score", ascending=False).tail(10)
    notop_10_2018 = df18.sort_values(by="Happiness Score", ascending=False).tail(10)
    notop_10_2019 = df19.sort_values(by="Happiness Score", ascending=False).tail(10)
    uno =go.Scatter(
                    y = notop_10_2015['Country'],
                    x = notop_10_2015['Happiness Score'],
                    mode = "markers",
                    name = "2015",
                    marker = dict(size= 10,color = 'red'),
                    text= notop_10_2015.Country)
# # creating trace2
# dos =go.Scatter(
#                     y = notop_5_2015['Country'],
#                     x = notop_5_2016['Happiness Score'],
#                     mode = "markers",
#                     name = "2016",
#                     marker = dict(size= 10,color = 'green'),
#                     text= notop_5_2015.Country)
# # creating trace3
# tres =go.Scatter(
#                     y = notop_5_2015['Country'],
#                     x = notop_5_2017['Happiness Score'],
#                     mode = "markers",
#                     name = "2017",
#                     marker = dict(size= 10,color = 'blue'),
#                     text= notop_5_2015.Country)

# # creating trace4
# cuatro =go.Scatter(
#                     y = notop_5_2015['Country'],
#                     x = notop_5_2018['Happiness Score'],
#                     mode = "markers",
#                     name = "2018",
#                     marker = dict(size= 10,color = 'black'),
#                     text= notop_5_2015.Country)

# creating trace5
    cinco =go.Scatter(
                    y = notop_10_2015['Country'],
                    x = notop_10_2019['Happiness Score'],
                    mode = "markers",
                    name = "2019",
                    marker = dict(size= 10,color = 'green'),
                    text= notop_10_2019.Country)


    data = [uno, cinco]
    layout = dict(title = "Evolución 2015 vs 2019 top 10 menos felices",
              xaxis= dict(title= 'Country',ticklen= 2,zeroline= False),
              yaxis= dict(title= 'Happiness',ticklen= 2,zeroline= False),
              hovermode="x unified"
             )
    fig = dict(data = data, layout = layout)
    st.plotly_chart(fig)

    st.write("")
    '''
    👉 En el 2015, el país menos feliz fue Togo con un puntaje de 2.84 mientras que en el 2019 fue Sudan con un puntaje de 2.85.
    Con estos dos países encontramos inconsistencias en cuanto a la diferencia de puntajes entre ambos años.

    👉 En el 2015 Sudan obtuvo un puntaje de 4.55 mientras que Togo en 2019 obtuvo 4.08. Veamos como se movieron sus factores explicativos:

    Economia 19 vs 15: Togo aumentó un 28% | Sudan bajó un 42%

    Familia 19 vs 15:  Togo aumentó un 300% | Sudan bajó un 43.56%

    Vida sana 19 vs 15: Togo aumentó un 46.43% | Sudan bajó un 21.62%

    👉Con esta información es dificil estimar que pasó en esos países, pero se puede decir que ambos tuvieron resultados opuestos e importantes
    lo que fomenta tales cambios en los puntajes.
    '''



    st.write("---")

    col1, col2, col3 = st.columns(3)
    if col2.button('GRACIAS!'):
        st.audio("ringtones-be-happy.mp3")


    








 