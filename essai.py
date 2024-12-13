class MLNaiveBayesClassifier(APrioriClassifier):    #qui utilise le maximum de vraisemblance (ML)
    # pour estimer la classe d'un individu en utilisant l'hypothèse du Naïve Bayes.
    def __init__(self,df):
        self.attrs = [col for col in df.columns if col != 'target']
        self.dic = dict() 
        for k in df.keys():
            if k != "target" and k != df.index.name:
                self.dic[k] = P2D_l(df,k) #Pour chaque attribut, on construit un dictionnaire de probabilité P(attr = a| target = t)
        super().__init__(df)
    
    def estimProbas(self, attrs):
        res0 = 1
        res1 = 1
        for i in attrs.keys():
            if i != "target":
                d = self.dic[i]
                if attrs[i] in d[1]:
                    res1 *= d[1][attrs[i]]
                else :
                    #Si la valeur n'est pas dans notre dictionnaire, la probabilité passe à 0
                    res1 = 0
                    break
                if attrs[i] in d[0]:
                    res0 *= d[0][attrs[i]]
                else :
                    #Si la valeur n'est pas dans notre dictionnaire, la probabilité passe à 0
                    res0 = 0
                    break
        return {0 : res0, 1 : res1}

    def estimClass(self, dico):
        d = self.estimProbas(dico)
        if d[0] > d[1] or np.abs(d[0]-d[1]) < 1e-15:
            return 0
        return 1

class MAPNaiveBayesClassifier(APrioriClassifier):
    def __init__(self,df):
        self.df = df
        self.dic = dict() 
        self.pt = getPrior(self.df)['estimation']   #p(target = 1)
        for k in df.keys():
            if k != "target":
                self.dic[k] = P2D_l(df,k)           #Pour chaque attribut, on construit un dictionnaire de probabilité P(attr = a| target = t)

    def estimProbas(self, attrs):
        res1 = self.pt      #p(target = 1)
        res0 = 1 - res1     #p(target = 0)
        for i in attrs:
            if i != "target" and i != self.df.index.name:
                d = self.dic[i]
                if attrs[i] in d[1]:
                    res1 *= d[1][attrs[i]]
                else :
                    res1 = 0
                    break
                if attrs[i] in d[0]:
                    res0 *= d[0][attrs[i]]
                else :
                    res0 = 0
                    break
        if res0 > 1e-15 or res1 > 1e-15:
            return {0 : res0/(res0+res1), 1 : res1/(res0+res1)}
        return {0 : 0, 1 : 0}

    def estimClass(self, dico):
        d = self.estimProbas(dico)
        if d[0] > d[1] or np.abs(d[0]-d[1]) < 1e-15:
            return 0
        return 1
    