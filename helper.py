import pandas as pd
import random

class Helper(object):
    """docstring for Helper"""
    def __init__(self):
        super(Helper, self).__init__()
        
    def get_dataset(self, path, hvdm = False, name = "cmc"):
     
        self.dataset = pd.read_csv(path, header=None)
        #self.dataset = self.dataset.drop(self.dataset.columns[0], axis=1)
        
        
    def return_accuracy(self, results):
        i = 0
        j = 0
        class_row = len(self.test_dataset.iloc[0])-1
        for result in results:
            if result == self.test_dataset.iloc[j][class_row]:
                i+=1
        success_rate = i / len(results)
        
        return success_rate

    def split_dataset(self):
        self.train = self.dataset.sample(frac=0.7, random_state= 200)
        test = self.dataset.drop(self.train.index)
        self.test_dataset = test #for accuracy purposes
        self.test = test.drop(test.columns[len(test.columns)-1], axis=1)
        return self.train, self.test




    def select_prototypes(self, classifier):
        #Implementing CNN for selecting prototypes
        
        self.S = self.train.sample(n=1)
        print(type(self.S))
        changed = True
        class_row = len(self.test.iloc[0])-1

        print(self.S.index)
        while changed:
            changed = False
            #print(self.S.index)
            for index, row in self.train.drop(self.S.index).iterrows():

                if row[class_row] != classifier.predict(row[0:len(row)-1]):

                    self.S = self.S.append(row)
                    changed = True
                    
                    #print(len(self.S)/len(self.train))
    
    """
    classifier: must be a 1-KNN classifier with trained data equal to S.
    """ 
    def LVQ1(self, classifier):
        P = self.S
        interations = 100
        class_row = len(self.train.iloc[0])-1
        dataset_values = P.values
       
        print("started LVQ")
        alpha = 0.99
       

        for i in range(0, interations):
            index = random.randrange(len(self.train)-1)
            x = self.train.iloc[index]
            #print("x: " + str(x))
            e = classifier.select(x)
            #print("e: " + str(e))
            #print(e[0][3])
            e_index = 0 
            for element in dataset_values:
                
                field_index = 0
                changed = False
              
                for value in e[0]:
                    if value != element[field_index]:
                        changed = True
                    field_index += 1            
                if not changed:
                    break
                else:
                    e_index += 1
           
            new_e = []
            
            if e[0][class_row] != x[class_row]:
                j = 0
                for field in e[0][0:class_row]:
                    new_e.append( field + alpha * (x[j] - field) )
                    j += 1
            else:
                j = 0
                for field in e[0][0:class_row]:
                    new_e.append( field + alpha * (x[j] - field) )
                    j += 1
            
            new_e.append(e[0][class_row])
            dataset_values[e_index] = new_e
            alpha = alpha * 0.9
        return pd.DataFrame.from_records(dataset_values)

    def LVQ2(self, classifier, distance_kind = "euclidean"):
        new_classifier = KNN(self.S, 1, distance_kind)
        P = self.LVQ1(new_classifier)
        dataset_values = P.values
        
        class_row = len(self.train.iloc[0])-1

        w = 0.5 #width

        interations = 50
        s = (1 - w)/(1 + w)
        alpha = 0.99
        print("started LVQ2")
        for i in range(0, interations):
            index =  random.randrange(len(dataset_values)-1)

            x = self.train.iloc[index]
            e_s = classifier.select(x)
            e_i, e_j = e_s[0], e_s[1] #get both nearest neighborhood
            e_i_distance = classifier.distance(x, e_i)
            e_j_distance = classifier.distance(x, e_j)

            if min(e_i_distance, e_j_distance) > s:
                new_e_i = []
                new_e_j = []
                print(e_i[0])
                if e_i[class_row] != e_j[class_row]:
                    if e_i[class_row] == x[class_row]:
                        j = 0
                        for field in e_i[0:class_row]:
                            new_e_i.append( field + alpha * (x[j] - field))
                            j += 1

                        j = 0
                        for field in e_j[0:class_row]:
                            new_e_j.append( field - alpha * (x[j] - field))
                            j += 1

                     
                    else:
                        j = 0
                        for field in e_i[0:class_row]:
                            new_e_i.append( field - alpha * (x[j] - field))
                            j += 1

                        j = 0
                        for field in e_j[0:class_row]:
                            new_e_j.append( field + alpha * (x[j] - field))
                            j += 1

                    #find their position to modify

                    e_i_index = 0 
                    for element in dataset_values:
                            
                        field_index = 0
                        changed = False
                          
                        for value in e_i:
                            if value != element[field_index]:
                                changed = True
                            field_index += 1            
                        if not changed:
                            break
                        else:
                            e_i_index += 1


                    e_j_index = 0 
                    for element in dataset_values:
                            
                        field_index = 0
                        changed = False
                      
                        for value in e_j:
                            if value != element[field_index]:
                                changed = True
                            field_index += 1            
                        if not changed:
                            break
                        else:
                            e_j_index += 1

                    new_e_i.append(e_i[class_row])
                    new_e_j.append(e_j[class_row])
                    dataset_values[e_j_index] = new_e_j
                    dataset_values[e_i_index] = new_e_i
            alpha = alpha * 0.9

        return pd.DataFrame.from_records(dataset_values)


    def LVQ3(self, classifier, distance_kind = "euclidean"):
        new_classifier = KNN(self.S, 1, distance_kind)
        P = self.LVQ1(new_classifier)
        dataset_values = P.values
        
        class_row = len(self.train.iloc[0])-1

        w = 0.5 #width

        interations = 50
        s = (1 - w)/(1 + w)
        alpha = 0.99
        fator_e = 0.5
        print("started LVQ3")
        for i in range(0, interations):
            index =  random.randrange(len(dataset_values)-1)

            x = self.train.iloc[index]
            e_s = classifier.select(x)
            e_i, e_j = e_s[0], e_s[1] #get both nearest neighborhood
            e_i_distance = classifier.distance(x, e_i)
            e_j_distance = classifier.distance(x, e_j)

            if min(e_i_distance, e_j_distance) > s:
                new_e_i = []
                new_e_j = []
               
                if e_i[class_row] != e_j[class_row]:
                    if e_i[class_row] == x[class_row]:
                        j = 0
                        for field in e_i[0:class_row]:
                            new_e_i.append( field + alpha * (x[j] - field))
                            j += 1

                        j = 0
                        for field in e_j[0:class_row]:
                            new_e_j.append( field - alpha * (x[j] - field))
                            j += 1

                    else:
                        j = 0
                        for field in e_i[0:class_row]:
                            new_e_i.append( field - alpha * (x[j] - field))
                            j += 1

                        j = 0
                        for field in e_j[0:class_row]:
                            new_e_j.append( field + alpha * (x[j] - field))
                            j += 1

                   

                
                elif e_i[class_row] == e_j[class_row] == x[class_row]:
                    j = 0
                    for field in e_i[0:class_row]:
                        new_e_i.append( field +  fator_e * alpha * (x[j] - field))
                        j += 1

                    j = 0
                    for field in e_j[0:class_row]:
                        new_e_j.append( field  + fator_e *alpha * (x[j] - field))
                        j += 1


                e_i_index = 0 
                for element in dataset_values:
                        
                    field_index = 0
                    changed = False
                      
                    for value in e_i:
                        if value != element[field_index]:
                            changed = True
                        field_index += 1            
                    if not changed:
                        print("found")
                        break
                    else:
                        e_i_index += 1


                e_j_index = 0 
                for element in dataset_values:
                        
                    field_index = 0
                    changed = False
                  
                    for value in e_j:
                        if value != element[field_index]:
                            changed = True
                        field_index += 1            
                    if not changed:
                        print("found")
                        break
                    else:
                        e_j_index += 1
            

                new_e_i.append(e_i[class_row])
                new_e_j.append(e_j[class_row])
                dataset_values[e_j_index] = new_e_j
                dataset_values[e_i_index] = new_e_i
                print("modified")
            alpha = alpha * 0.9

        return pd.DataFrame.from_records(dataset_values)