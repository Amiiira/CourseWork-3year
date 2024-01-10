import json
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, spearmanr
import pandas as pd
import statsmodels.api as sm
import numpy as np

def social_media(data):
    social_media_counts = {}

    for participant_data in data:
        for question, answer in participant_data:
            if question.startswith("В каких социальных сетях Вы сидите ?") and answer:
                social_media_counts[answer] = social_media_counts.get(answer, 0) + 1

    social_media = list(social_media_counts.keys())
    counts = list(social_media_counts.values())

    bar_color = '#C7B9D6'
    plt.bar(social_media, counts, color=bar_color)
    plt.ylabel('Количество участников')
    plt.title('Выбор социальных сетей')
    plt.show()

def type_of_content(data):
    content_counts = {}

    for participant_data in data:
        for question, answer in participant_data:
            if question == "Какой контент Вы чаще всего потребляете ?" and answer:
                content_counts[answer] = content_counts.get(answer, 0) + 1

    content_categories = list(content_counts.keys())
    counts = list(content_counts.values())

    colors = ['#C7B9D6', '#B0A1C7', '#988DB8', '#8078A9', '#6F6A99']

    plt.pie(counts, labels=content_categories, colors=colors, autopct='%1.1f%%', startangle=180)
    plt.axis('equal')  # Задаем равные осям для кругового графика
    plt.title('Потребляемый контент')
    plt.show()

def hours_spend(data):
    hours_counts = {}

    for participant_data in data:
        for question, answer in participant_data:
            if question == "Какое среднее количество часов в день вы обычно проводите в социальных сетях?" and answer:
                hours_counts[answer] = hours_counts.get(answer, 0) + 1

    hours_categories = list(hours_counts.keys())
    counts = list(hours_counts.values())

    colors = ['#C7B9D6', '#B0A1C7', '#988DB8', '#8078A9', '#6F6A99']
    plt.pie(counts, labels=hours_categories, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')  # Задаем равные осям для кругового графика
    plt.title('Среднее количество часов в соц.сетях в день')
    plt.show()

def detox(data):
    detox_counts = {}

    for participant_data in data:
        for question, answer in participant_data:
            if question == 'Делаете ли Вы "детокс" от социальных сетей ? Хотя бы на несколько часов' and answer:
                detox_counts[answer] = detox_counts.get(answer, 0) + 1

    detox_categories = list(detox_counts.keys())
    counts = list(detox_counts.values())

    colors = ['#C7B9D6', '#B0A1C7', '#988DB8', '#8078A9', '#6F6A99']

    plt.pie(counts, labels=detox_categories, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')  # Задаем равные осям для кругового графика
    plt.title('Делаете ли Вы "детокс" от соц.сетей?')

    plt.show()

def wellbeing(data):
    emotional_counts = {}
    mental_counts = {}
    physical_counts = {}

    # Debugging print statements
    for participant_data in data:
        for question, answer in participant_data:
            if question == 'Каково ваше самочувствие после проведенного времени в социальных сетях? / Эмоциональное состояние' and answer:
                emotional_counts[answer] = emotional_counts.get(answer, 0) + 1
            elif question == 'Каково ваше самочувствие после проведенного времени в социальных сетях? / Психическое состояние' and answer:
                mental_counts[answer] = mental_counts.get(answer, 0) + 1
            elif question == 'Каково ваше самочувствие после проведенного времени в социальных сетях? / Физическое состояние' and answer:
                physical_counts[answer] = physical_counts.get(answer, 0) + 1

    emotional_categories = list(emotional_counts.keys())
    emotional_counts_values = list(emotional_counts.values())

    mental_categories = list(mental_counts.keys())
    mental_counts_values = list(mental_counts.values())

    physical_categories = list(physical_counts.keys())
    physical_counts_values = list(physical_counts.values())

    bar_color = '#C7B9D6'

    # Эмоциональное состояние
    plt.subplot(3, 1, 1)
    plt.barh(emotional_categories, emotional_counts_values, color=bar_color)
    plt.xlabel('Количество участников')
    plt.title('Эмоциональное состояние')

    # Психическое состояние
    plt.subplot(3, 1, 2)
    plt.barh(mental_categories, mental_counts_values, color=bar_color)
    plt.xlabel('Количество участников')
    plt.title('Психическое состояние')

    # Физическое состояние
    plt.subplot(3, 1, 3)
    plt.barh(physical_categories, physical_counts_values, color=bar_color)
    plt.xlabel('Количество участников')
    plt.title('Физическое состояние')

    plt.tight_layout()
    plt.show()

def reduce_time(data):
    reduce_time_counts = {}
    reduce_time_amount_counts = {}

    for participant_data in data:
        for question, answer in participant_data:
            if question == 'Хотели бы Вы проводить меньше времени в соц.сетях ?' and answer:
                reduce_time_counts[answer] = reduce_time_counts.get(answer, 0) + 1
            elif question == 'Если на прошлый вопрос Вы ответили утвердительно, то на сколько бы вы уменьшили время препровождения в соц.сетях ?' and answer:
                reduce_time_amount_counts[answer] = reduce_time_amount_counts.get(answer, 0) + 1

    reduce_time_categories = list(reduce_time_counts.keys())
    reduce_time_counts_values = list(reduce_time_counts.values())

    reduce_time_amount_categories = list(reduce_time_amount_counts.keys())
    reduce_time_amount_counts_values = list(reduce_time_amount_counts.values())

    colors = ['#C7B9D6', '#B0A1C7', '#988DB8', '#8078A9', '#6F6A99']

    # Pie chart for the desire to reduce time spent on social media
    plt.subplot(2, 1, 1)
    plt.pie(reduce_time_counts_values, labels=reduce_time_categories, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('Хотели бы Вы проводить меньше времени в соц.сетях ?')

    # Pie chart for the amount of time participants would reduce if desired
    plt.subplot(2, 1, 2)
    plt.pie(reduce_time_amount_counts_values, labels=reduce_time_amount_categories, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('Если да, то на сколько ?')

    plt.tight_layout()
    plt.show()

def stress(data):
    stress_counts = {}

    for participant_data in data:
        for question, answer in participant_data:
            if question == 'Бывали ли случаи, когда вы испытывали стресс из-за событий в социальных сетях?' and answer:
                stress_counts[answer] = stress_counts.get(answer, 0) + 1

    stress_categories = list(stress_counts.keys())
    stress_counts_values = list(stress_counts.values())

    colors = ['#C7B9D6', '#B0A1C7', '#988DB8', '#8078A9', '#6F6A99']

    plt.pie(stress_counts_values, labels=stress_categories, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('Вы испытывали стресс из-за событий в социальных сетях?')

    plt.show()

def chi_analyses(data):
    # Подготовка данных для теста Хи-квадрат
    reduce_time_stress_data = {'Да_Да': 0, 'Да_Нет': 0, 'Нет_Да': 0, 'Нет_Нет': 0}

    for participant_data in data:
        reduce_time = next(answer for question, answer in participant_data 
                        if question == 'Хотели бы Вы проводить меньше времени в соц.сетях ?')
        stress = next(answer for question, answer in participant_data 
                        if question == 'Бывали ли случаи, когда вы испытывали стресс из-за событий в социальных сетях?')

        key = f"{reduce_time}_{stress}"
        reduce_time_stress_data[key] += 1

    # Проведение теста Хи-квадрат
    observed_values = [
        [reduce_time_stress_data['Да_Да'], reduce_time_stress_data['Да_Нет']],
        [reduce_time_stress_data['Нет_Да'], reduce_time_stress_data['Нет_Нет']]
    ]

    chi2, p, _, _ = chi2_contingency(observed_values)

    # Вывод результатов теста
    print(f"Хи-квадрат: {chi2}")
    print(f"P-value: {p}")

    # Интерпретация результатов
    if p < 0.05:
        print("Есть статистически значимая связь.")
    else:
        print("Статистической значимости связи не выявлено.")


def spreamnan(data):
    # Извлекаем ответы для каждой группы для вопросов о социальных сетях и времени
    group1_social_media_data = [
        item[1] for item in data[0] if "социальных сетях" in item[0] or "Делаете ли Вы" in item[0]
    ]
    group2_social_media_data = [
        item[1] for item in data[1] if "социальных сетях" in item[0] or "Делаете ли Вы" in item[0]
    ]

    # Преобразуем категории в числовые значения (предполагая, что "больше 3 часов" = 1, "меньше 3 часов" = 0)
    category_mapping = {"больше 3 часов": 1, "меньше 3 часов": 0}
    group1_numeric_data = [category_mapping.get(answer, answer) for answer in group1_social_media_data]
    group2_numeric_data = [category_mapping.get(answer, answer) for answer in group2_social_media_data]

    # Вычисляем коэффициент корреляции Спирмена
    correlation_coefficient, _ = spearmanr(group1_numeric_data, group2_numeric_data)

    print(f"Коэффициент корреляции Спирмена для вопросов о социальных сетях и времени: {correlation_coefficient}")

def spearman_2(data):
    stress_data = ["Да", "Нет", "Да", "Да", "Нет"]  # Пример данных по стрессу
    time_spent_data = ["Да", "Да", "Нет", "Нет", "Да"]  # Пример данных по времени в соц.сетях

    # Преобразование к числовым значениям
    numeric_stress_data = [1 if answer == "Да" else 0 for answer in stress_data]
    numeric_time_spent_data = [1 if answer == "Да" else 0 for answer in time_spent_data]

    # Рассчитываем коэффициент корреляции Спирмена
    spearman_corr, _ = spearmanr(numeric_stress_data, numeric_time_spent_data)

    print(f"Коэффициент корреляции Спирмена: {spearman_corr}")

def v_kramer(data):
    # Создаем DataFrame
    df = pd.DataFrame([dict(item) for item in data])

    # Пересчитываем категориальные переменные в бинарные значения
    df['Детокс'] = (df['Делаете ли Вы "детокс" от социальных сетей ? Хотя бы на несколько часов'] == 'Да').astype(int)
    df['Меньше времени'] = (df['Хотели бы Вы проводить меньше времени в соц.сетях ?'] == 'Да').astype(int)

    # Убираем лишние столбцы
    df = df[['Детокс', 'Меньше времени']]

    # Создаем таблицу сопряженности
    contingency_table = pd.crosstab(df['Детокс'], df['Меньше времени'])

    # Рассчитываем коэффициент V Крамера
    chi2, _, _, _ = chi2_contingency(contingency_table)
    n = contingency_table.sum().sum()
    min_dim = min(contingency_table.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * min_dim)) if (n * min_dim) != 0 else 0

    print(f"Коэффициент V Крамера между детоксом и желанием проводить меньше времени в соц.сетях: {cramers_v}")

def v_kramer_2(data):
    df = pd.DataFrame([dict(item) for item in data])

    # Пересчитываем категориальные переменные в бинарные значения
    df['Стресс'] = (df['Бывали ли случаи, когда вы испытывали стресс из-за событий в социальных сетях?'] == 'Да').astype(int)
    df['Меньше времени'] = (df['Хотели бы Вы проводить меньше времени в соц.сетях ?'] == 'Да').astype(int)

    # Убираем лишние столбцы
    df = df[['Стресс', 'Меньше времени']]

    # Создаем таблицу сопряженности
    contingency_table = pd.crosstab(df['Стресс'], df['Меньше времени'])

    # Рассчитываем коэффициент V Крамера
    chi2, _, _, _ = chi2_contingency(contingency_table)
    n = contingency_table.sum().sum()
    min_dim = min(contingency_table.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * min_dim)) if (n * min_dim) != 0 else 0

    print(f"Коэффициент V Крамера между стрессом и желанием проводить меньше времени в соц.сетях: {cramers_v}")


if __name__ == '__main__':
    with open('survey.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # detox(data)
    # social_media(data)
    # type_of_content(data)
    # hours_spend(data)
    # wellbeing(data)
    # reduce_time(data)
    # stress(data)

    # chi_analyses(data)

    # spreamnan(data)
    # spearman_2(data)
    
    v_kramer(data)
    v_kramer_2(data)


