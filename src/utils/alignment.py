from collections import Counter

class Levenshtein:
    @staticmethod
    def lev_dist(pred_text, ref_text):
        past_states = set()       # keeps track of all states visited
        frontier = [((0,0), '')]  # states with cost 'dist' to visit in this loop 
        next_frontier = []        # states to visit in next step
        dist = 0                  # begin by searching all paths 0 away

        if isinstance(pred_text, list) and all(type(i) == str for i in pred_text):
            pred_text = [i.lower() for i in pred_text if isinstance(i, str)]
            ref_text  = [i.lower() for i in ref_text if isinstance(i, str)]
        
        while True:
            for (k1, k2), path in frontier:
                # If either text has reached the end
                if k1 == len(pred_text) or k2 == len(ref_text):
                    # If reached the end state, return current (best) path
                    if k1 == len(pred_text) and k2 == len(ref_text):
                        finished = True
                        return (dist, path)
                    # Otherwise only add the next possible state
                    elif k1 == len(pred_text):
                        next_frontier.append( ((k1, k2+1), path+'i') )
                    elif k2 == len(ref_text):
                        next_frontier.append( ((k1+1, k2), path+'d') )
                # If first elements are the same, add next state to this step
                elif pred_text[k1] == ref_text[k2]:
                    frontier.append(((k1+1, k2+1), path+'_'))
                # Else consider to delete, insert or replace the first element
                else:
                    next = [((k1+1, k2), 'd'), ((k1, k2+1), 'i'), ((k1+1, k2+1), 'r')]
                    for state, dec in next:
                        # Only consider states not visited
                        if state not in past_states:
                            next_frontier.append((state, path+dec))
                            past_states.add(state)

            frontier = next_frontier    # for next iteration update frontier
            next_frontier = []          # reset the next frontier
            dist += 1                   # all next paths have a larger cost

    @staticmethod
    def lev_dist_dynamic(pred_text, ref_text):
        def levenshtine(k1, k2):
            #If state visited before, load cost
            if (k1, k2) in memo:
                dist, path = memo[(k1, k2)]
                return (dist, path)
            #base case 1: if ref string empty can only delete all
            elif k1 == len(pred_text):
                dist = len(ref_text) - k2
                path = 'i'*dist
            #base case 2: if input string empty can only insert all
            elif k2 == len(ref_text):
                dist = len(pred_text) - k1
                path = 'd'*dist
            # If first elements are the same, move to next elements (cost is same)
            elif pred_text[k1] == ref_text[k2]:
                dist, path = levenshtine(k1+1, k2+1)
                path = '_'+path
            #otherwise recursively select the best possible decision
            else:
                dist_1, path_1 = levenshtine(k1+1, k2)
                dist_2, path_2 = levenshtine(k1  , k2+1)
                dist_3, path_3 = levenshtine(k1+1, k2+1)
                path_1, path_2, path_3 = 'd'+path_1, 'i'+path_2, 'r'+path_3
                dist, path = min((dist_1+1, path_1), (dist_2+1, path_2), (dist_3+1, path_3))

            # add state to dict of visited states
            memo[(k1, k2)] = (dist, path)
            return (dist, path)

        memo = {}                       #initilise the memoized data
        dist, path = levenshtine(0, 0)  #begin algorithm at the start
        return (dist, path)
    
    @classmethod
    def align_text(cls, pred_text, ref_text):
        dist, path = cls.lev_dist(pred_text, ref_text)
        pred_text, ref_text = pred_text.copy(), ref_text.copy()
        aligned = ['']
        
        for step in path:
            if step in ['_', 'r']:
                dec, _ = pred_text.pop(0), ref_text.pop(0)
                aligned.append(dec)
            elif step == 'd':
                dec = pred_text.pop(0)
                aligned[-1] += (' '+dec)
            elif step == 'i':
                dec = ref_text.pop(0)
                aligned.append('')
        aligned[0] += ' ' + aligned.pop(1)
        aligned[0] = aligned[0].strip()
        return aligned   

    @classmethod
    def wer(cls, pred_text, ref_text, report=False)->tuple:        
        dist, path = cls.lev_dist(pred_text, ref_text)

        err_counts = Counter(path)
        rep, ins, dels = (err_counts[i] for i in ['r', 'i', 'd'])
        if report:
            print(f"WER:{dist/len(ref_text):.3f}  replace:{rep/len(ref_text):.3f}  ",
                  f"inserts: {ins/len(ref_text):.3f}  deletion: {dels/len(ref_text):.3f}")
        return (dist, rep, ins, dels)

    @classmethod
    def detailed_errors(cls, pred_text, ref_text)->list:
        dist, path = cls.lev_dist(pred_text, ref_text)
        pred_text, ref_text = pred_text.copy(), ref_text.copy()
        
        errors = []
        for step in path:
            if step == '_':
                pred_text.pop(0), ref_text.pop(0)
            elif step == 'r':
                pred_word, ref_word = pred_text.pop(0), ref_text.pop(0)
                errors.append(('r', pred_word, ref_word))
            elif step == 'd':
                err_word = pred_text.pop(0)
                errors.append(('d', err_word, None))
            elif step == 'i':
                err_word = ref_text.pop(0)
                errors.append(('i', None, err_word))
        return errors   
