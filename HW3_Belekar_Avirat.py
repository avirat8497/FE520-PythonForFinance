"""Name : Avirat Belekar
    Id: 10454332

"""
class MovingAverage:

    def __init__(self, size):
        self.size = size # initilize the window size to size
        self.queue = []  # initilized a queue

    def next(self, val):
       if not self.queue or len(self.queue) < self.size:
           self.queue.append(val) # Only if there 1 element to be added in the queue
       else:
           self.queue.pop(0)
           self.queue.append(val) # append the value in the queue
       return float(sum(self.queue)) / len(self.queue) # return the moving average

    
class subway:

    def __init__(self):
        self.checkin = {}
        self.times = {}

    def checkIn(self, id, stationName, t): # checkin function
        # add your code here
        self.checkin[id] = (stationName, t)

    def checkOut(self, id, stationName, t): # checkout function
        try:
            self.times[self.checkin[id][0] + "-" + stationName].append(t - self.checkin[id][1])
        except KeyError:
            self.times[self.checkin[id][0] + "-" + stationName] = [t - self.checkin[id][1]]

    def getAverageTime(self, startStation, endStation): # function for average time at a station
        return sum(self.times[startStation + "-" + endStation]) / len(self.times[startStation + "-" + endStation])

class Linear_regression:

    def __init__(self, x, y, m, c, epochs, L): # initilize variables
        self.x = x
        self.y = y
        self.m = m
        self.c = c
        self.epochs = epochs
        self.L = L
        
        
    def gradient_descent(self) : # calculate gradient descent
        for i in range(self.epochs):
            diff_y = []
            Dm = []
            y_pred = []
            sum_dm = 0
            sum_dc = 0
            for i  in range(len(self.x)):
                d = (self.x[i][0] * self.m) + self.c
                y_pred.append(d)
                diff_y.append(y_pred[i] - self.y[i])
                Dm.append(self.x[i][0] * diff_y[i])
                sum_dm = sum_dm + Dm[i]
                sum_dc = sum_dc + diff_y[i]
            avg_dm = (sum_dm) / len(Dm)
            avg_dc = sum_dc / len(diff_y)
            self.m = self.m - self.L * avg_dm
            self.c = self.c - self.L * avg_dc
        return self.m, self.c

    
    def predict(self,x_new):
        y_pred_new = []
        for i in range(len(x_new)):
            y_new = self.c + self.m * x_new[i] # calculating prediction values using linear regression (y = mx + c)
            y_pred_new.append(y_new)
        return y_pred_new

    
    
class LCG:

    def __init__(self, seed, multiplier, increment, modulus):
        self.seed = seed
        self.multiplier = multiplier
        self.increment = increment
        self.modulus = modulus

        
    def get_seed(self):   
        return self.seed
         
    def set_seed(self,new_seed):   
        self.seed = new_seed
        return self.seed
    
    def initialize(self):
       return self.seed
    
    def gen(self): # generating the next random number using the formula x1 = (a*x0 + c) mod y where x1 = new random number,a = multiplier,x0 = seed,c = increment,y = modulus
        self.seed = (((self.multiplier * self.seed) + self.increment) % self.modulus)
        return self.seed / self.modulus
        
    def seq(self, num): # generating a sequence
        return [self.gen() for i in range(0,num)]

if __name__ == "__main__":



    x = [[0.18], [1.0], [0.92], [0.07], [0.85], [0.99], [0.87]]
    y = [109.85, 155.72, 137.66, 76.17, 139.75, 162.6, 151.77]
    x_new = [0.9,0.8,0.40,0.7]

    
    # Test Question 1
    print("\nQ1")

    windowsize = 3 
    moving_average = MovingAverage(windowsize)
    step1 = moving_average.next(1)  
    print("my answer: ", step1)    
    print("right answer: 1.0")    
    print("--------------")
    step2 = moving_average.next(10) 
    print("my answer: ", step2)    
    print("right answer: 5.5")    
    print("--------------")  
    step3 = moving_average.next(3) 
    print("my answer: ", step3)    
    print("right answer: 4.66667")    
    print("--------------") 
    step4 = moving_average.next(5) 
    print("my answer: ", step4)    
    print("right answer: 6.0")    
    print("--------------") 
    
    
    
    # Test Question 2
    print("\nQ2") 
    s = subway()
    s.checkIn(10,'Leyton',3)
    s.checkOut(10,'Paradise',8)
    print("my answer: ",s.getAverageTime('Leyton','Paradise'))
    print("right answer: 5.0")    
    print("--------------") 
    s.checkIn(10,'Leyton',10)
    s.checkOut(10,'Paradise',16)
    print("my answer: ",s.getAverageTime('Leyton','Paradise'))
    print("right answer: 5.5")    
    print("--------------") 
    s.checkIn(10,'Leyton',21)
    s.checkOut(10,'Paradise',30)
    print("my answer: ",s.getAverageTime('Leyton','Paradise'))
    print("right answer: 6.667")    
    print("--------------") 
    
    
    # Test Question 3
    print("\nQ3") 
    Linear_model = Linear_regression(x,y,0,0,500,0.001)
    print("I use m=0, c=0, epochs=500, L=0.001")
    print("my m and c: ",Linear_model.gradient_descent())
    print("right m and c:(35.97890301691016, 46.54235227399102)")    
    print("--------------") 
    print("my predict: ", Linear_model.predict(x_new))
    print(" right predict: [78.92336498921017, 75.32547468751915, 60.93391348075509, 71.72758438582812]")
    
    
    # Bonus Question 
    print("\nBonus") 
    print("set seed = 1, multiplier = 1103515245, increment = 12345, modulus = 2**32")
    lcg = LCG(1,1103515245,12345, 2**32 )
    print("my seed is: ", lcg.get_seed())
    print("right seed is: 1")
    print("the seed is setted with: ", lcg.set_seed(5))
    print("right seed is setted with 5")
    print("the LCG is initialized with seed: ",lcg.initialize())
    print("the LCG is initialized with seed 5")
    print("the next random number is: ", lcg.gen())
    print("right next random number is: 0.2846636981703341")
    print("the first ten sequence is: ", lcg.seq(10))
    print("the first ten sequence is: ", [0.6290451611857861, 0.16200014390051365, 0.4864134492818266, 0.655532845761627, 0.8961918593849987, 0.2762452410534024, 0.8611323081422597, 0.9970241007395089, 0.798466683132574, 0.46138259768486023])


