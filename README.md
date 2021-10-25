# Курсовая работа за 2 семестр по динамике полета

![py ver](https://img.shields.io/badge/Python-3.9%2B-blue)

## Использование

1. Установите модули из requirements.txt

	pip install -r path\to\requirements.txt

2. Запустите run.py

3. В окне tkinter с названием "Coursework" введите вариант в поле "Вариант:" и выберите папку сохранения.

4. Вsберите формат сохранения графиков.

5. Запустите кнопкой "Запустить"

## Соответствие результатов с методическим указанием

Результаты таблиц будут сохранены в папке "RESULTS". 
Порядок и пункты будут в соответствии с методическими указаниями. 
Ссылка[здесь](https://disk.yandex.ru/i/hzpGRXPHZ-vW9g)   

**Пункт 2.2 Расчет летно-технических характеристик самолета**

Результаты таблицы 1 по высотам в файлах 'alt_0.0.csv', 'alt_2.0.csv' и т.д, где цифры высота в Км. Порядок колонок соответствует с таблицей 1.\

График 1 - C_H=0.0\
График 2 - P_H=0.0\
График 3 - V_y_H=0.0\
График 4 - q_ch_H=0.0\
График 5 - q_km_H=0.0

Результаты таблицы 2 в файле 'table_2.csv'

График 1 - H_M_flight_area\
График 2 - V_y_H\
График 3 - q_km_q_ch


**Пункт 2.3.1 Расчет характеристик набора высоты**

Таблица 3 в файле 'climb_data.csv'.\
Обобщение результатов в файле 'climb_mini_table.csv'.\

График 1 - climb_params\
График 2 - H_climb

**Пункт 2.3.2 Расчет характеристик крейсерского полета**

По результатам расчетов формируется таблица 'level_flight_data.csv'

**Пункт 2.3.3 Расчет характеристик участка снижения**

Таблица 3 в файле 'descent_data.csv'.\
Обобщение результатов в файле 'descent_mini_table.csv'.\

График 1 - descent_params\
График 2 - H_des\
Сомещенный график для участков набора высоты, крейсерского полета и снижения:\
График 3 - H_L_graph\

**Пункт 2.4 Расчет диаграммы транспортных возможностей неманевренного самолета**

Таблице 5 соответствует 'cargo_load.csv'\

График 1 - m_L_graph

**Пункт 2.5 Расчет взлетно-посадочных характеристик самолета**

По результатам расчетов формируется таблица 'takeoff_landing_table.csv'

**Пункт 2.6 Расчет характеристик маневренности самолета**

Таблица 7 в файле 'turn_data_table.csv'\

График 1 - turn_graph

**Пункт 2.7 Расчет характеристик продольной статической устойчивости и управляемости**

Для определения потребной площади ГО строится график с названием 'xTP_graph.'\

Таблица 10 - 'otn_S_go.csv'\
Таблица 11 - 'sigmas_table.csv'\
Таблица 12 - 'phi_table_H=0.CSV'\

График 1 - phi_bal_graph\
График 2 - phi_n_graph\
График 3 - ny_p_graph\

На этом все расчеты заканчиваются. 

## Необходимые дополнительные модули

	matplotlib==3.4.3
	numpy==1.21.3
	Pillow==8.4.0
	scipy==1.7.1
	data_handler_csv==0.0.2

[Git link: data_handler_csv](https://github.com/lalapopa/data_handler_csv)









