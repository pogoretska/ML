import argparse

parser = argparse.ArgumentParser(description = "select best 500 programs using criteria")

parser.add_argument("--input", help = "path to .zip archive with .csv files", required = True)
parser.add_argument("--price-symbol", help = "set a price symbol (@...)", required = True)
parser.add_argument("--algorithm", help = "min-opens: minimize open positions, min-bargains: minimize bargains, max-profit: maximize avg bargain's profit", required = True)
parser.add_argument("--weight", help = "set weight of first criterion, 0 <= w <= 1", type = float, default = 0.7)
parser.add_argument("-v", "--verbose", help="print debug messages", action="store_true")
args = parser.parse_args()

def extract(line):
    if line[-1] == "\n": return line[:-1].split(";")
    return line.split(";")

import zipfile

 #start with csv file opening
if not zipfile.is_zipfile( args.input ): exit("Invalid path to .zip archive")
else:
    zf = zipfile.ZipFile( args.input, 'r')
    fileName = args.price_symbol + "_programs.csv"
    if not fileName in zf.namelist(): exit("There's no file '{}' in a given archive".format(fileName))
    else:
        fcsv = zf.open( fileName )
        header = extract( fcsv.readline() )
        rows = []
        for line in fcsv:
            rows.append(dict(zip(header, extract(line))))
        zf.close()

from operator import add
from random import sample, random, randint
import numpy

 #make an array of all programs,
 #each row = [id of program = row number from csv file (int), fsymbol (string), iprofit (float), insample (int32 array)]
 #while minimizing open positions we count the number of open positions for each string and store it instead of actual string content
allData = []
count = 0
posToNum = { "L" : 1, "0" : 0, "S" : -1 }

if args.algorithm == "min-opens":
    for row in rows:
        allData.append([ count, row['fsymbol'], float(row['iprofit']), len(row['insample'].replace('0',''))])
        count += 1
else:
    for row in rows:
        allData.append([ count, row['fsymbol'], float(row['iprofit']), numpy.array([ posToNum[foo] for foo in row['insample'] ], dtype = numpy.int32)])
        count += 1                       
                       
 #generate an individual consisting of random set of programs from all programs array, while count defines cardinality of a set 
def individual(count):  
    return sample(allData, count)

 #generate population with given parameters - (how many individuals in population, how many programs contains each individual)
def population(howMany, howManyEach):
    return [ individual(howManyEach) for x in xrange(howMany) ]     #in our case each individ consists of 500 random programs

 #how many distinct features contains given individ
def count_features(individ):     
    return len(set([ x[1] for x in individ ]))

 #average distinct features per population
def count_avg_features(pop):
    return 1.0*reduce(add, [count_features(x) for x in pop]) / len(pop)

 #how many open positions contains given ansamble
def count_opens(individ):
    return reduce(add, [ x[3] for x in individ])
    
 #count bargains for given individ
def count_bargains(individ):
     # calculating sum of insample arrays
    flattened = reduce(add, [ cell[3] for cell in individ ])
     
     # count changes in a resulting vector of each individ         
    count = 0
    opened = 0
    for x in flattened:
        count += abs(opened - x)
        opened = x
    return int( count / 2 )

 #count average profit per bargain
def count_profit(individ):
    profit = reduce(add, [ float(x[2]) for x in individ ])
    bargains = count_bargains(individ)
    return 1.0*profit / bargains

 #additional function to show how good is individ by 2nd criterion
def count_2nd(individ, method):
    if method == 'min-opens': return count_opens(individ)
    elif method == 'min-bargans': return count_bargains(individ)
    elif method == 'max-profit': return count_profit(individ)

 # optimize by 2 criteria with weights; weight is given for 1st criterion - minimizing number of distinct features to be 49 or less
 # weights are such that w1 + w2 = 1, i.e. w2 = 1 - w1
 # formulae is F(x) = w1 * ( 1 - individ_score_by_criteria_1 / optimal_score_1 )**2 + w2 * ( 1 - individ_score_by_criteria_2 / optimal_score_2 )**2
def fitness(individ, criterion, weight):
    if criterion == 'min-opens': fitScore = weight * (1 - 1.0*count_features(individ)/49) ** 2 + (1 - weight) * (1 - 1.0*count_opens(individ)/1000000) ** 2
    elif criterion == 'min-bargains': fitScore = weight * (1 - 1.0*count_features(individ)/49) ** 2 + (1 - weight) * (1 - 1.0*count_bargains(individ)/100000) ** 2
    elif criterion == 'max-profit': fitScore = weight * (1 - 1.0*count_features(individ)/49) ** 2 + (1 - weight) * (1 - 1.0*count_profit(individ)/1000) ** 2
    return fitScore

def evolve(pop, method, count, retain = 0.2, randomSelect = 0.05, mutate=0.01):
    gradedFit = [ (fitness(x, method, args.weight), x) for x in pop ]
    graded = [ x[1] for x in sorted(gradedFit) ]

     #in verbose mode let's print results
    if args.verbose and not count % 10:
        print "Cycle " + str(count)
        print "Best element fitness {}, distinct features {}, {} criterion {}".format(fitness(graded[0], method, args.weight), count_features(graded[0]), args.algorithm, count_2nd(graded[0], args.algorithm))
        print "Avg features per generation: " + str(count_avg_features(pop))
    
    retainLength = int(len(graded) * retain)

     # select best 20%
    parents = graded[:retainLength]

     # randomly add other individuals to promote genetic diversity
    for individual in graded[retainLength:]:
        if randomSelect > random():
            parents.append(individual)

     # mutate some individuals
    for individual in parents:
        if mutate > random():
            posToMutate = randint(0, len(individual)-1)
             # we select random program from list of all programs to replace some position in chosen individual
            individual[posToMutate] = allData[randint(0, len(allData) - 1)]

     # crossover parents to create children
    parentsLength = len(parents)
    desiredLength = len(pop) - parentsLength
    children = []
    while len(children) < desiredLength:
        male = randint(0, parentsLength - 1)
        female = randint(0, parentsLength - 1)
        if male != female:
            male = parents[male]
            female = parents[female]
            half = len(male) / 2
            child = male[:half] + female[half:]
            children.append(child)

    parents.extend(children)
    return parents

if args.algorithm == 'min-opens':
    if args.verbose: print 'Using min-opens'
elif args.algorithm == 'min-bargains':
    if args.verbose: print 'Using min-bargains'
elif args.algorithm == 'max-profit':
    if args.verbose: print 'Using max-profit'
else: exit("Cannot recognize algorithm name. Use one of (min-opens, min-bargains, max-profit)")

 #population(how many individuals in population, how many programs contains each individual)
p = population(500, 500)

 #let's run!
# cycles = 10000
#print "Element with best fitness"
#for i in xrange(cycles):
#    p = evolve(p, args.algorithm, i)

i = 0
while count_avg_features(p) > 49:
    p = evolve(p, args.algorithm, i)
    i += 1

gradedFit = [ (fitness(x, args.algorithm, args.weight), x) for x in p ]
graded = sorted(gradedFit)

if args.verbose:
    print "Final {}th population best element's results (avg population features {})".format(i, count_avg_features(p))
    print "Fitness {}, distinct features {}, 2nd criterion {}".format(graded[0][0], count_features(graded[0][1]), count_2nd(graded[0][1], args.algorithm))

resFile = file('results-{}.csv'.format(args.algorithm),'w')
resFile.write(';'.join(header) + '\n')
for x in graded[0][1]:
    line = ';'.join([ rows[x[0] ][key] for key in header]) + '\n'
    resFile.write(line)
        