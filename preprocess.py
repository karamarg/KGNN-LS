import argparse
import numpy as np

RATING_FILE_NAME = dict({'movie': 'ratings.csv', 'book': 'BX-Book-Ratings.csv', 'music': 'user_artists.dat'})
SEP = dict({'movie': ',', 'book': ';', 'music': '\t'})
THRESHOLD = dict({'movie': 4, 'book': 0, 'music': 0})

# This function reads and processes a file that maps item indices to entity IDs, and it creates mappings between these IDs and their corresponding indices, stored in the item_index_old2new and entity_id2index dictionaries.
# final output:  [item_index | entity_id]
def read_item_index_to_entity_id_file():
    file = 'data/' + DATASET + '/item_index2entity_id.txt' #'../data/' + DATASET + '/item_index2entity_id.txt' Constructs the path of the item index to entity ID mapping file
    print('reading item index to entity id file: ' + file + ' ...')
    i = 0 #This variable is used for indexing
    for line in open(file, encoding='utf-8').readlines(): # reads the file's content line by line
        item_index = line.strip().split('\t')[0] # removes leading and trailing whitespace characters, splits the line into a list of strings and selects the first element which is assumed to be the item index
        satori_id = line.strip().split('\t')[1] # extracts the second element from the list, which is assumed to be the Satori ID
        item_index_old2new[item_index] = i #This line updates a dictionary named item_index_old2new. It uses the item_index as the key and assigns the value i to it. This is likely used to map old item indexes to new ones.
        entity_id2index[satori_id] = i #This line updates a dictionary named entity_id2index. It uses the satori_id as the key and assigns the value i to it. This is likely used to map entity IDs to their corresponding indices.
        i += 1


# It creates a dataset with all the users and the entities where they had given positive ratings. For each user extra rows (equal to the positive ratings) of randomly selected negative samples are drown
# from the pool of entities that had not been rated by the user. Final output: [userID | item_index | target(0/1)]
def convert_rating():
    file = 'data/' + DATASET + '/' + RATING_FILE_NAME[DATASET] #'../data/' + DATASET + '/' + RATING_FILE_NAME[DATASET] Constructs a file path by combining the string 'data/' with the value of the variable DATASET and the filename specified in the RATING_FILE_NAME dictionary for the given dataset.

    print('reading rating file ...')
    item_set = set(item_index_old2new.values()) # creates a set called item_set that contains unique values from the item_index_old2new dictionary (the values not the keys). It collects all unique item indices.
    user_pos_ratings = dict() #These two lines create two dictionaries to store positive and negative user ratings. These dictionaries will map user IDs to sets of items.
    user_neg_ratings = dict()

    for line in open(file, encoding='utf-8').readlines()[1:]: #It then iterates through the lines in the file, skipping the first line which is the header
        array = line.strip().split(SEP[DATASET]) #reads a line from the file, removes leading and trailing whitespace characters, and splits it into a list of strings using the delimiter specified in the SEP dictionary for the given dataset.

        # remove prefix and suffix quotation marks for BX dataset
        if DATASET == 'book':
            array = list(map(lambda x: x[1:-1], array)) # the map() function executes a specified function for each item in an iterable

        # in lines 37-44, it keeps only the lines where the artist ids exist in the item indices of the item_index2entity_id.tx file and for each artistID it keeps the new item_index created with the read_item_index_to_entity_id_file function
        item_index_old = array[1] #extracts the item index from the array.
        if item_index_old not in item_index_old2new:  # the item is not in the final item set
            continue # if the item index is not in the item_index_old2new dictionary, it skips to the next iteration of the loop. This is likely used to filter out items that are not in the final item set
        item_index = item_index_old2new[item_index_old] #maps the old item index to a new item index using the item_index_old2new dictionary.

        user_index_old = int(array[0]) # extracts the user index (converting it to an integer) from the array.

        rating = float(array[2]) #  extracts the rating (as a floating-point number) from the array.

        # in lines 47-54 creates a positive and a negative dictionary where the keys are the userIDs and their corresponding value is a list with all the new item indices of the artistIDs i.e. {2: {32, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31}, 3: {33}, 4: {24, 34, 35}}
        if rating >= THRESHOLD[DATASET]: # checks if the rating is greater than or equal to a threshold value specified in the THRESHOLD dictionary for the given dataset.
            if user_index_old not in user_pos_ratings: # it adds the user_index_old to the user_pos_ratings dictionary, creating a set for the user if it doesn't exist, and adds the item_index to the set.
                user_pos_ratings[user_index_old] = set()
            user_pos_ratings[user_index_old].add(item_index)
        else:
            if user_index_old not in user_neg_ratings:
                user_neg_ratings[user_index_old] = set()
            user_neg_ratings[user_index_old].add(item_index)

    print('converting rating file ...')
    writer = open('data/' + DATASET + '/ratings_final.txt', 'w', encoding='utf-8') #../ opens a new file for writing, named 'ratings_final.txt'
    user_cnt = 0
    user_index_old2new = dict()
    # in lines 61-65 creates new indices fot the userIDs in the positive ratings dictionary
    for user_index_old, pos_item_set in user_pos_ratings.items(): # iterates through the users in the user_pos_ratings dictionary and their associated positive item sets.
        if user_index_old not in user_index_old2new: # checks if the old user index is not already in the user_index_old2new dictionary.
            user_index_old2new[user_index_old] = user_cnt # If the user index is not in the dictionary, it adds the user index to the user_index_old2new dictionary and assigns a new index (user_cnt) to it.
            user_cnt += 1 #  increments user_cnt to ensure that the new user indices are unique.
        user_index = user_index_old2new[user_index_old] #obtains the new user index from the user_index_old2new dictionary.

        for item in pos_item_set: # iterates through the positive item set associated with the user.
            writer.write('%d\t%d\t1\n' % (user_index, item)) #For each user-item pair, it writes a line to the 'ratings_final.txt' file in the format "user_index\titem\t1". This is likely to represent a positive rating (1).
        unwatched_set = item_set - pos_item_set # calculates a set of unwatched items by taking the difference between the item_set (all items) and the pos_item_set (items the user has positively rated).
        if user_index_old in user_neg_ratings: #checks if the user has negative ratings in the user_neg_ratings dictionary.
            unwatched_set -= user_neg_ratings[user_index_old] #If there are negative ratings, it removes items in the user_neg_ratings set from the unwatched_set.
        for item in np.random.choice(list(unwatched_set), size=len(pos_item_set), replace=False): # it randomly selects items from the unwatched_set using NumPy's random.choice function. The number of items selected is equal to the length of the pos_item_set, and items are not replaced after selection (replace=False).
            writer.write('%d\t%d\t0\n' % (user_index, item)) #For each user-item pair, it writes a line to the 'ratings_final.txt' file in the format "user_index\titem\t0". This is likely to represent a negative rating (0).
    writer.close() #closes the 'ratings_final.txt' file.
    print('number of users: %d' % user_cnt)
    print('number of items: %d' % len(item_set)) #number of unique items


#converts the knowledge graph (KG) file to the final KG where the entities  have been mapped to satori indexes and the relations to numerical indices.
#final output: [head|relation|tail] depicted as integers
def convert_kg():
    print('converting kg file ...')
    entity_cnt = len(entity_id2index) #calculates the number of entities by getting the length of the entity_id2index dictionary
    relation_cnt = 0

    writer = open('data/' + DATASET + '/kg_final.txt', 'w', encoding='utf-8') #opens a file named 'kg_final.txt' in the 'data/DATASET/' directory for writing
    for line in open('data/' + DATASET + '/kg.txt', encoding='utf-8'): #opens another file named 'kg.txt' in the 'data/DATASET/' directory for reading
        array = line.strip().split('\t') #reads a line from the file, removes leading and trailing whitespace characters, and splits it into a list of strings using the tab character ('\t') as the delimiter. The resulting array should contain three elements: head, relation, and tail.
        head_old = array[0]
        relation_old = array[1]
        tail_old = array[2]

        # in lines 92-100 the head and tails that are not already in the satori ids, are added to the entity_id2index dictionary as keys and a new value is generated
        if head_old not in entity_id2index: #checks if head_old is not already in the entity_id2index dictionary, meaning it's a new entity.
            entity_id2index[head_old] = entity_cnt #If it's a new entity, it adds head_old to the entity_id2index dictionary and assigns it a new index (entity_cnt). This is likely used to create an index for the entity if it doesn't exist already.
            entity_cnt += 1
        head = entity_id2index[head_old] #retrieves the index associated with head_old from the entity_id2index dictionary and assigns it to the variable head.

        if tail_old not in entity_id2index: #checks if tail_old is not already in the entity_id2index dictionary, indicating a new entity.
            entity_id2index[tail_old] = entity_cnt #If it's a new entity, it adds tail_old to the entity_id2index dictionary and assigns it a new index (entity_cnt).
            entity_cnt += 1
        tail = entity_id2index[tail_old] #retrieves the index associated with tail_old from the entity_id2index dictionary and assigns it to the variable tail.

        if relation_old not in relation_id2index: # checks if relation_old is not already in the relation_id2index dictionary, indicating a new relation.
            relation_id2index[relation_old] = relation_cnt #If it's a new relation, it adds relation_old to the relation_id2index dictionary and assigns it a new index (relation_cnt). This is likely used to create an index for the relation if it doesn't exist already.
            relation_cnt += 1
        relation = relation_id2index[relation_old] #retrieves the index associated with relation_old from the relation_id2index dictionary and assigns it to the variable relation.

        writer.write('%d\t%d\t%d\n' % (head, relation, tail)) #For each line in the 'kg.txt' file, this line writes a new line to the 'kg_final.txt' file with the format "head\trelation\ttail," where head, relation, and tail are the indices for the entities and relation.

    writer.close() #closes the 'kg_final.txt' file.
    print('number of entities (containing items): %d' % entity_cnt)
    print('number of relations: %d' % relation_cnt)


if __name__ == '__main__': #This line checks if the script is being run as the main program, rather than being imported as a module into another script. The code within this block will execute if the script is the main entry point.
    np.random.seed(555)

    parser = argparse.ArgumentParser() #creates an ArgumentParser object named parser, which is used for parsing command-line arguments and options.
    parser.add_argument('-d', type=str, default='movie', help='which dataset to preprocess') #adds a command-line argument definition to the parser. It specifies that the script expects an argument with the flag -d followed by a string.
    args = parser.parse_args() #parses the command-line arguments provided when the script is run and stores the results in the args variable. In this case, it will include the value specified with the -d flag, representing the dataset to preprocess.
    DATASET = args.d # assigns the value of the -d argument to the DATASET variable, which is used to determine which dataset to preprocess.

    entity_id2index = dict() #initialize three dictionaries: entity_id2index, relation_id2index, and item_index_old2new. These dictionaries are intended to store mappings between IDs and indices for entities, relations, and items.
    relation_id2index = dict()
    item_index_old2new = dict()

    read_item_index_to_entity_id_file()
    convert_rating()
    convert_kg()

    print('done')
