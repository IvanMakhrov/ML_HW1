Вначале мы провели первичный EDA - выявили размера тренировочного и тестового датасетов - (6999 и 1000 строк соответственно). В каждлом датасете 13 
признаков
Увидели, что в тестовом датасете есть пропуски в столбцах mileage, engine, max_power, torque и seats
Также, в тестовом датасете были обнаружены дубликаты

Далее, мы построили отчет по тестовому датасету с помощью ydata_profiling, итоги анализа следующие:
В данных для обучения 6999 строк, присутствуют пропуски и дубликаты

В поле name отсутствуют пропущенные значения, 1924 уникальных объекта
В датасете представлены данные по АМ, которые были выпущены в период 1983-2020 гг. Для каждой АМ указан год выпуска
Цена продажи варьирует от 29999 до 10000000, пропуски отсутствуют
Средний пробег АМ - 69585 км. Для каждой АМ указан пробег
Для каждой АМ указан тип топлива, которых всего 4 (Diesel, Petrol, CNG, LPG). Доля дизельных АМ превышает долю бензиновых АМ. Для каждой АМ указан тип 
топлива
Для каждой АМ указан тип продавца. Доля частных лиц значимо превышает долю дилеров
Для каждой АМ указан тип корбки передач. Доля ручной коробки передач превышает долю автоматической коробки передач
Доля каждой АМ указано кол-во владельцев. Наибольшая доля АМ с одним владельцем
Есть пропуски данных в поле с расходом топлива, которые отображаются как кол-во км на 1 л
Не для каждой АМ указан объем двигателя, мощность и крутящий момент
Также есть пропуски в поле с количеством сидений

Можно увидеть сильную прямую взаимосвязь между следующими признаками:

Ценой продажи и годом выпуска АМ
Ценой продажи и трансмиссией
Можно увидеть сильную обратную взаимосвязь между следующими признаками:

Год выпуска АМ и пробег
Год выпуска АМ и цена продажи
В датасете присутствует 493 уникальных строки с дубликатами
Всего дубликатов в датасете - 985 строк

Анализируем статистики датасета и получаем следующие выводы:
Различия между средней и медианой внутри датасета говорят об отклонении от нормального распределения
В train и test наибольшее отклонение от нормального распределения в поле km driven
Значимых отличий в среднем и медиане между train и test нет. Это говорит о том, что разделение данных на train и test проведено корректно и можно 
ожидать, что оценки качества модели и ошибки на train и test будут похожими

Затем, мы удаляем дубликаты из тестового датасета
Далее, мы корректируем оба датасета - тестовый и тренировочный
Мы убираем единицы измерения из mileage, engine и max_power, приводим их к типу float
Предобрабатываем torque - делим на два признака torque и max_torque_rpm

Далее, мы заполняем пропуски в обоих датасетах медианными значениями вещественных признаков из тренировочного датасета

Анализируем как изменилось распределение тренировочного датасета после наших изменений:
Поле seats:
Было: Среднее: 5,41, Отклонение: 0,97
Стало: Среднее: 5,87, Отклонение: 2,57
В случае наличия значимого количества пропусков и заполнения медианой распределение сдвинется в сторону нормального распределения. Также это может 
уменьшить среднее отклонение. В нашем случае оно увеличилось из-за удаления дублей
В случае заполнения средним значением, распределение сдвинется в сторону выбросов

Далее, мы строим pairplot и делаем следующие выводы на их основании:
На основе графиков можно увидеть что происходит с целевой переменной при изменении одного из признаков
Например, можно заметить, что при уменьшении года выпуска АМ, ее стоимость снижается
Также можно заметить, что при увеличении max_power увеличивается стоимость АМ

На трейне незначительно отличается распределение на графике между стоимостью АМ и пробегом АМ
В остальном, распределения на трейне и тесте похожи

Затем, мы смотрим на коррееляцию между признаками и делаем следующие выводы:
Наименьшая корреляция наблюдается между признаками:

max_power и km_driven
max_torque_rpm и year
Сильная положительная связь наблюдается между:

engine и max_power
engine и torque
max_power и torque
Между годом выпуска и пробегом существует умеренная обратная связь. Это говорит о том, для части данных будет справедливо утверждение, что при 
уменьшении года выпуска АМ увеличивается пробег АМ, но говорить точно мы об этом не можем

Затем, мы смотрим различия между коэффициентами корреляции Пирсона и Спирмена
Можно увидеть, что есть различия между корреляциями Пирсона и Спирмена
Например, рассмотрим влияние показателей на целевую переменную:

Корреляция Спирмена > корреляции Пирсона для year, torque, seats. Следовательно, между показателями существует нелинейная связь

Также, мы исследуем тестовые данные на выбросы, строим boxplotпо вещественным признакам
Исследуем данные на выбросы, построим boxplot по каждой числовой переменной
Видим, что по переменым year, selling_price, mileage, engine, max_power есть выбросы, которые могут повлиять на качество модели
seats мы не рассматриваем, так как у него мало уникальных значений и признак может быть отнесен к категориальным

После анализа данных, мы переходим к построению моделей

Строим модель линейной регрессии и смотрим ее результаты
По модели обученной на трейне видно, что она объясняет 59,6% дисперсии
Для теста r^2 отличается не сильно. Это ознаначает, что модель не переобучена и данные для трейна и теста разделены корректно
По значению MSE сложно сделать выводы о качестве модели

Также мы анализируем отличия между R^2 и R^2 - adjusted
R^2 - adjusted применяется в многофакторной регрессии для оценки влияния признаков на дисперсию
Например, если при добавлении нового признака R^2 adjusted снизилась, то данный признак не вносит значимого вклада в модель
Таким образом, можно оставить в датасете только значимые признаки

После этого, мы стандартизируем признаки с помощью StandatdScaler и вновь строим модель линейной регрессии
Метрики MSE и R^2 практически не изменились

На основании анализа коэффициентов модели можно сделать вывод о том, что наиболее информативным признаком оказался max_power, он вносит наибольший вклад 
в целевую переменную

Далее, мы применяем регуляризацию - строим Лассо-регрессию
MSE и R^2 вновь не изменились
L1 регуляризация не занулила коэффициенты. Следовательно, каждый из показателей вносит значимый вклад в объяснение дисперсии

Далее, с помощью gridSearchCV 10 фолдами подбираем оптимальные гиперпараметры для модели Лассо-регрессии
GridSearch обучил alpha*cv = 100 моделей
Каждый параметр в param_grid - это список гиперпараметров, которые мы применяем к модели
В нашем случае, alpha - это константа, которая умножается на штраф и увеличивает силу регуляризации (больше alpha - больше сила регуляризации)
Лучший коэффцициент регуляризации - alpha=1000. Веса показателей не занулились

Затем, с помощью GridSearchCV мы находим оптимальные гиперпараметры для модели ElasticNet
GridSearch обучил alpha*l1_ratio*cv = 1100 моделей
Лучшая модель - это модель Лассо регрессии (l1_ratio=1) с alpha = 1000

Модель с L0 регуляризацией построить не удалось

Далее, мы работаем с категориальными фичами
Мы предобрабатываем столбец name, в котором содержатся данные о марке, модели и комплектации
В связи с большой вариативностью описания внутри пораметра name, удалось корректно вытащить только марку и модель АМ

Мы кодируем категориальные параметры с помощью OneHotEncoding
В процессе OHE мы преобразуем категориальные признаки в бинарные. Количество уникальных элементов категориальных признаков не должно быть очень большим
Мы используем n-1 столбцов, так как если брать n столбцов, то будет возникать линейная зависимость
Например, если у категории Grade есть значения: Good, Nice и Great, то после OHE мы получаем 2 столбца (например, Grade_Good, Great_Nice). В случае, 
если Grade_Good = False и Grade_Nice = False, то мы понимаем, что поле Grade_Great имело бы значение True, несмотря на то, что в явном виде оно 
отсутствует
Мы можем удалить незначимые признаки

После этого, мы строим модель с L2 регуляризацией - Ridge-регрессию с учетом OHE
Получаем R^2=0.89, что является хорошим показателем

После различных преобразований датасета и применения различных моделей мы можем построить одну, которая будет давать лучший прогноз

Мы преобразуем признаки torque, mileage, engine, max_power, приводя их к числовому типу
Извлекаем из поля name марку и модель АМ
Далее, мы генерируем новые признаки на основе уже существующих, а именно:
Возраст АМ
Средний пробег в год
Мощность на объем двигателя
Квадрат года
Категория возраста АМ
Категория АМ по мощности

После этого, мы обрабатываем числовые признаки:
Заполняем пропуски медианой
Масштабируем с помощью StandardScaler

Обрабатываем категориальные признаки:
Заполняем пропуски значением NA
Делаем OneHotEncoding

После этого строим ElasticNet с alpha=0.0001 и l1_ratio=0.9
На тестовых данных получаем средующий результат:
r2: 0.9202955882333865
MSE: 45816371893.814255

То есть, модель объясняет 92.03% дисперсии

Построим для модели бизнес метрики:
Считаем долю АМ, у которых прогнозная цена отличается от реальной цены менее, чем на 10%
Получаем, что для 32.4% АМ прогнозная цена отличается от реальной менее, чем на 10%

Также, построим метрику, которая будет отражать среднее отклонение между прогнозной и реальной ценами
В случае, если прогнозная цена ниже реальной, то в таком случае штрафовать будем в 2 раза больше

Напишем api с помощью FastApi
Реализуем два сценария:
1.Отправляем json с информацией об одном АМ и получаем прогнозное значение цены
2.Отправляем csv файл с данными об АМ и получаем обратно csv файл с еще одной колонкой - прогнозной ценой АМ

Тестирование первого сценария:
![Alt text](https://github.com/IvanMakhrov/ML_HW1/blob/c96ccb6632e4f98ae811a5ce52c21862ce724fc5/images/predict_item_postman.png?raw=true)

![Alt text](https://github.com/IvanMakhrov/ML_HW1/blob/main/images/predict_item_python.png?raw=true)

![Alt text](https://github.com/IvanMakhrov/ML_HW1/blob/main/images/predict_items_postman.png?raw=true)
