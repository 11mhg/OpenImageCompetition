

class Tree(object):
    def __init__(self,hierarchy_dict=None):
        self.hierarchy_dict = hierarchy_dict
        self.root_node = Node(name='root',level=0)
        self.parse_level(sub=self.hierarchy_dict,parent=self.root_node)

    def parse_level(self,sub=None,parent=None):
        if 'LabelName' in sub:
            ind = parent.add_child(sub['LabelName'])
        if 'Subcategory' in sub:
            l = sub['Subcategory']
            for elem in l:
                self.parse_level(elem,parent.children[ind]) 
    
    def max_level(self,n):
        if n.leaf:
            return n.level
        values = []
        for c in n.children:
            values.append(self.max_level(c))
        return max(values)

    def get_num_level(self,n=None,level=0):
        if not n:
            n=self.root_node.children[0]
        s = 0
        if n.level <= level:
            s+=1
        for child in n.children:
            s+= self.get_num_level(child,level)
        return s

    def get_class_list(self,n=None,level=0):
        if not n:
            n = self.root_node.children[0]
        if n.level <= level:
            cl = [n.name]
        else:
            cl = []
        for child in n.children:
            cl += self.get_class_list(child,level)
        return cl
    
class Node(object):
    def __init__(self,name=None,level=0):
        self.name=name
        self.level=level
        self.children=[]
        self.leaf = True 

    def add_child(self,child_name):
        self.leaf = False
        node = Node(name=child_name,level=self.level+1)
        self.children.append(node)
        return len(self.children)-1
