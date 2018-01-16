import re

ROOT_PATH = './data/'
FILENAME = 'nomen.sql'
NEUTRAL = 'neutral.txt'
FEM = 'feminine.txt'
MASC = 'masculine.txt'

TRAINING = ROOT_PATH + 'training/'
VALIDATION = ROOT_PATH + 'validation/'

with open(ROOT_PATH + FILENAME, 'r') as all_words, \
    open(TRAINING + NEUTRAL, 'w') as training_neutrals, \
    open(TRAINING + MASC, 'w') as training_masculines, \
    open(TRAINING + FEM, 'w') as training_feminine, \
    open(VALIDATION + NEUTRAL, 'w') as validation_neutrals, \
    open(VALIDATION + MASC, 'w') as validation_masculines, \
    open(VALIDATION + FEM, 'w') as validation_feminine:

    read_data = all_words.read()

    m = re.finditer('\(.*?\)', read_data)

    count_masculine = 0
    count_feminine = 0
    count_neutral = 0

    for match in m:
        splitted = match.group().split(',')[2:4]

        if len(splitted) > 1:
            word = splitted[0].strip('\'') + '\n'

            if splitted[1] == '\'SUB:NOM:SIN:MAS\'':
                count_masculine += 1

                if count_masculine % 20 == 0:
                    validation_masculines.write(word)
                else:
                    training_masculines.write(word)

            if splitted[1] == '\'SUB:NOM:SIN:FEM\'':
                count_feminine += 1

                if count_feminine % 20 == 0:
                    validation_feminine.write(word)
                else:
                    training_feminine.write(word)

            if splitted[1] == '\'SUB:NOM:SIN:NEU\'':
                count_neutral += 1

                if count_neutral % 20 == 0:
                    validation_neutrals.write(word)
                else:
                    training_neutrals.write(word)
