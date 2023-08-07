import random

import matplotlib.pyplot as plt
import numpy as np
from enum import Enum

import logging

from alpha_trees.EdgeQueue import EdgeQueue, Edge
#from charting import map_globally

BOTTOM = -1

class CC_TYPE(Enum):
    BCKGRND = 0
    FRGRND = 1
    ALL = 2

def differences_per_pixel_global(model, orig, patch_sz,channels=3):
    img = np.copy(orig)
    # container for patches' labels
    hor = np.ones((img.shape[0], img.shape[1] - 1)) * -1
    vert = np.ones((img.shape[0] - 1, img.shape[1])) * -1
    img = np.pad(img, ((patch_sz // 2, patch_sz // 2), (patch_sz // 2, patch_sz // 2), (0, 0)), mode='symmetric')
    patches = np.lib.stride_tricks.sliding_window_view(img, window_shape=(patch_sz, patch_sz), axis=(0, 1))
    print(np.shape(patches))
    patches_orig = patches.reshape(-1, patch_sz * patch_sz * channels)

    #patches=patches_orig
    labels, patches = model.predict(patches_orig, return_projected=True)
    for x in range(0, orig.shape[1] - 1, 1):
                p1 = np.array([patches[orig.shape[0] * y + x] for y in range(orig.shape[0])])
                p2 = np.array([patches[orig.shape[0] * y + x + 1] for y in range(orig.shape[0])]) # TODO check
                hor[:, x] = np.sum((p1 - p2)**2, 1)
    for y in range(0, orig.shape[0] - 1, 1):
                p1 = np.array([patches[orig.shape[1] * y + x] for x in range(orig.shape[1])])
                p2 = np.array([patches[orig.shape[1]* (y + 1) + x] for x in range(orig.shape[1])])
                vert[y,:] = np.sum((p1 - p2)**2, 1)

    return hor, vert,patches_orig,labels



def differences_per_pixel(model, orig, patch_sz, labels,channels=3, inf_value=np.inf):
    img = np.copy(orig)
    classes = model.classes_
    # container for patches' labels
    hor = np.ones((img.shape[0], img.shape[1]- 1)) * inf_value
    vert = np.ones((img.shape[0] - 1, img.shape[1])) * inf_value
    img = np.pad(img, ((patch_sz // 2, patch_sz // 2), (patch_sz // 2, patch_sz // 2), (0, 0)), mode='symmetric')
    patches = np.lib.stride_tricks.sliding_window_view(img, window_shape=(patch_sz, patch_sz), axis=(0, 1))
    print(np.shape(patches))
    patches = patches.reshape(-1, patch_sz*patch_sz*channels)

    for x in range(0, labels.shape[1] - 1, 1):
        for c in classes:  # horizontal diff
            cidx = np.intersect1d(np.where(labels[:, x] == c)[0], np.where(labels[:, x + 1] == c)[0])
            if cidx.size:
                p1 = np.array([patches[labels.shape[0] * y + x] for y in cidx])
                p2 = np.array([patches[labels.shape[0] * y + x + 1] for y in cidx]) # TODO check
                hor[cidx, x] = model.distance(p1, p2, model.omegas_[c])
    for y in range(0, labels.shape[0] - 1, 1):
        for c in classes:  # verical diff
            cidx = np.intersect1d(np.where(labels[y, :] == c)[0], np.where(labels[y + 1, :] == c)[0])
            if cidx.size:
                p1 = np.array([patches[labels.shape[1] * y + x] for x in cidx])
                p2 = np.array([patches[labels.shape[1]* (y + 1) + x] for x in cidx])
                vert[y,cidx] = model.distance(p1, p2, model.omegas_[c])
    return hor, vert,patches

class Node:
    def __init__(self, parent=None, alpha=0,  area=0,patch_sz=1, channels=3):
        self.parent = parent  # parent index
        self.alpha = alpha
        self.label = None
        self.area = area
        self.node_indices = set()
        self.features = np.zeros(patch_sz*patch_sz*channels)
        self.cls=np.zeros(3)



    def normalize_features(self):
        self.features = self.features/ self.area


class AlphaTree:


    def __init__(self, img,patch_sz):
        self.img = img
        img_sz =  img.shape[0] * img.shape[1]
        self.patch_sz=patch_sz
        self.nodes = [Node(patch_sz=patch_sz) for _ in range(2*img_sz)]
        self.roots = [0] * 2 * img_sz
        self.edgeq = EdgeQueue()
        self.current_sz = img_sz


    def new_node(self,  alpha):
        self.nodes[self.current_sz] = Node(alpha=alpha, parent=BOTTOM, patch_sz=self.patch_sz)
        self.roots[self.current_sz] = BOTTOM
        new_idx = self.current_sz
        self.current_sz += 1
        return new_idx

    def update_node_features(self, p, q=None):
        x, y = p % self.img.shape[1], p // self.img.shape[1]
        if q is None: # called in make_set
            self.nodes[p].features= self.patches[self.img[0].shape[0] * y + x]
            self.nodes[p].cls[self.labels[self.img[0].shape[0] * y + x]] += 1 # cast a vote
        else:
            self.nodes[p].features += self.nodes[q].features
            self.nodes[p].cls += self.nodes[q].cls





    def phase1(self, height, width,  lmbda, alphas_h, alphas_v):
        """
        Create an alpha tree based on the provided alpha values.
     these alpha values are either combined in the salience tree or
     they are stored as edges in the edge queue.
        :param height: height of an image/mask
        :param width: width of an image/mask
        :param alphas_mask_h:  horizontal alpha values
        :param alphas_mask_v: vertical alpha values
        :param lmbda:threshold to determine if we have encountered an edge
        :return:
        """
        # process first row
        for x in range(1, width):
            self.make_set(x)
            edge_salience = alphas_h[0, x - 1]
            if edge_salience < lmbda:  # connected
                self.union1(x, x - 1)
            else: # store the edge
                self.edgeq.push(Edge(x, x-1, edge_salience))

        # for all other rows
        for y in range(1,height):
            p = y*width # p is the first pixel in a row
            self.make_set(p)
            edge_salience = alphas_v[y-1, 0]

            if edge_salience < lmbda:
                self.union1(p, p-width)
            else:
                self.edgeq.push(Edge(p, p-width, edge_salience))
            p += 1
            # for each column in the current row
            for x in range(1, width):
                # repeat process in y-direction
                self.make_set(p)
                edge_salience = alphas_v[y-1, x]

                if edge_salience < lmbda:
                    self.union1(p,p-width)
                else:
                    self.edgeq.push(Edge(p, p - width, edge_salience))
                # repeat process in x-direction
                edge_salience = alphas_h[y, x - 1]

                if edge_salience < lmbda:
                    self.union1(p, p - 1)
                else:
                    self.edgeq.push(Edge(p, p - 1, edge_salience))
                p += 1




    def phase2(self):
        """
         Create additional nodes in the tree based on the collected edges in phase1 and construct the final tree.
        :return:
        """
        while not self.edgeq.empty():
            # deque the current edge and temporarily store its values
            current_edge = self.edgeq.pop()
            current_edge.p, current_edge.q = self.ancestors(current_edge.p, current_edge.q)
            if current_edge.p != current_edge.q:
                if current_edge.p < current_edge.q:
                    current_edge.p, current_edge.q = current_edge.q, current_edge.p
                if self.nodes[current_edge.p].alpha < current_edge.alpha:
                    # if the higher node has a lower alpha level than the edge
                    # we combine the two nodes in a new salience node
                    r = self.new_node(current_edge.alpha)
                    self.union2(r, current_edge.p)
                    self.union2(r, current_edge.q,)
                else:
                    # otherwise we add the lower node to the higher node
                    self.union2(current_edge.p, current_edge.q)



    def phase2_preset_alphas(self, alphas):
        """
         Create additional nodes in the tree based on the collected edges in phase1 and construct the final tree.

        :return:
        """
        alphas = np.concatenate(([0], alphas, [np.inf]))
        alphas = np.sort(np.unique(alphas))
        i=0
        while not self.edgeq.empty() and i < len(alphas):
            current_edge = self.edgeq.pop()
            if  current_edge.alpha <= alphas[i]:
                current_edge.p, current_edge.q = self.ancestors(current_edge.p, current_edge.q)
                if current_edge.p != current_edge.q:
                    if current_edge.p < current_edge.q:
                        current_edge.p, current_edge.q = current_edge.q, current_edge.p

                    if self.nodes[current_edge.p].alpha < current_edge.alpha:
                        # if the higher node has a lower alpha level than the edge
                        # we combine the two nodes in a new salience node
                        r = self.new_node(alphas[i])
                        self.union2(r, current_edge.p)
                        self.union2(r, current_edge.q,)
                    else:
                        # otherwise we add the lower node to the higher node
                        self.union2(current_edge.p, current_edge.q)
            else:
                i +=1
        if len(self.edgeq.queue) > 0:
            raise ValueError("Level queue is non empty")




    def make_set(self, p):
        # happens for each pixel once
        self.nodes[p].parent = BOTTOM
        self.nodes[p].alpha = 0
        self.nodes[p].area = 1
        self.nodes[p].node_indices.add(p)
        self.update_node_features(p)
        self.roots[p] = BOTTOM


    def union1(self, p, q):
        """
        Combines the regions of two pixels
        :param p: "current" pixel
        :param q: neighbor pixel
        :return:
        """
        q = self.find_root1(q)
        if q != p:
            # set p to be q's parent
            self.nodes[q].parent = p
            self.roots[q] = p
            self.nodes[p].area = self.nodes[p].area + self.nodes[q].area
            self.nodes[p].node_indices.add(q)
            self.update_node_features(p, q)



    def union2(self, p, q):
        self.nodes[q].parent = p
        self.roots[q] = p
        self.nodes[p].area += self.nodes[q].area
        self.update_node_features(p, q)
        self.nodes[p].node_indices.add(q)



    def find_root1(self, p):
        r = p
        # find the root of a tree and set it to r
        while self.roots[r] != BOTTOM:
            r = self.roots[r]
        i = p
        # r=root; i= current pixel; invariant: current pixel != root
        while i != r:
            j = self.roots[i]
            self.roots[i] = r
            self.nodes[i].parent = r # change parent in th tree to total root
            i = j # i becomes its own root
        return r


    def level_root(self, p):
        """
        Finds the root node of the alpha level of a given node.
        :param p: Node to find the level root of
        :return: Index of the level root
        """
        r = p
        # find the root of the node
        while not self.is_level_root(r):
            r = self.nodes[r].parent

        i = p
        # add node to the branch
        while i != r:
            j = self.nodes[i].parent
            self.nodes[i].parent = r
            i=j

        return r

    def is_level_root(self, i):
        """
        Determine if a node at a given index is the root of its level of the tree.
  A node is considered the level root if it has the different alpha level as its parent
  or does not have a parent.
        :param i: index of a node
        :return:
        """
        parent = self.nodes[i].parent
        if parent == BOTTOM:
            return True
        return self.nodes[i].alpha != self.nodes[parent].alpha

    def find_root(self, p):
        """
        Find the root of a given node p.
        :param p:  Node of which we want to find the root
        :return: The index of the root
        """
        r = p
        while self.roots[r] != BOTTOM:
            r= self.roots[r]
        i = p
        while i != r:
            j = self.roots[i]
            self.roots[i] = r
            i=j
        return  r
    def ancestors(self, p ,q):
        # get root of each pixel and ensure correct order
        p = self.level_root(p)
        q = self.level_root(q)

        if p < q:
            p, q = q, p
        # while both nodes are not the same and are not the root of the tree
        while p != q and self.roots[p] != BOTTOM and self.roots[q] != BOTTOM:

            q = self.roots[q]
            if p < q:
                p, q = q, p
        # if either node is the tree root find the root of the other

        if self.roots[p] == BOTTOM:
            q = self.find_root(q)
        elif self.roots[q] == BOTTOM:
            p = self.find_root(p)
        return p,q



    def build(self, alphas, model, labels=None, alpha_start=0):
        if labels is None:
            alphas_h, alphas_v, self.patches, self.labels = differences_per_pixel_global(model, self.img, self.patch_sz)
        else:
            alphas_h, alphas_v,self.patches = differences_per_pixel(model, self.img, self.patch_sz, labels)
        # root is a separate case
        self.make_set(0)

        # Phase 1 combines nodes that are not seen as edges and fills the edge queue with found edges
        self.phase1(self.img.shape[0], self.img.shape[1], alpha_start, alphas_h, alphas_v)
        # Phase 2 runs over all edges, creates SalienceNodes

        if len(alphas) > 0:
            self.phase2_preset_alphas(alphas)
        else:
            self.phase2()


        del self.edgeq
        del self.roots
        self.nodes = self.nodes[:self.current_sz]

        # r,g,b, features are set to mean by dividing by the final node area
        for i in range(0, self.current_sz):
            self.nodes[i].normalize_features()



    def filter(self, lmbda):
        """
        :param lmbda: difference between pixels is at most lmbda. Large lambda -> less details
        :return:
        """
        if lmbda <= self.nodes[self.current_sz-1].alpha:
            #   Set the output value of the root
            self.nodes[self.current_sz-1].label=0
            # Set colours of the other nodes
            label = 1
            for i in range(self.current_sz-2, -1,-1):
                if self.is_level_root(i) and self.nodes[i].alpha >= lmbda:
                    # set the color of the level root
                    self.nodes[i].label = label
                    label += 1
                else:
                    # use parents' color
                    self.nodes[i].label = self.nodes[self.nodes[i].parent].label
        else:
            # set to black
            for i in range(0, self.current_sz):
                self.nodes[i].label = -1
        result = np.zeros((self.img.shape[0], self.img.shape[1]))
        for i in range(self.img.shape[0] * self.img.shape[1]):
            result[i//self.img.shape[1], i % self.img.shape[1]] = self.nodes[i].label
        return result


    def filter2(self, lmbda):
        """
        :param lmbda: difference between pixels is at most lmbda. Large lambda -> less details
        :return:
        """
        if lmbda <= self.nodes[self.current_sz-1].alpha:
            #   Set the output value of the root
            self.nodes[self.current_sz-1].label=self.nodes[self.current_sz-1].alpha
            # Set colours of the other nodes

            for i in range(self.current_sz-2, -1,-1):
                if self.is_level_root(i) and self.nodes[i].alpha >= lmbda:
                    # set the color of the level root
                    self.nodes[i].label = self.nodes[i].alpha

                else:
                    # use parents' color
                    self.nodes[i].label = self.nodes[self.nodes[i].parent].label
        else:
            # set to black
            for i in range(0, self.current_sz):
                self.nodes[i].label = -1
        result = np.zeros((self.img.shape[0], self.img.shape[1]))
        for i in range(self.img.shape[0] * self.img.shape[1]):
            result[i//self.img.shape[1], i % self.img.shape[1]] = self.nodes[i].label
        return result



    def filter3(self, lmbda):
        """
        :param lmbda: difference between pixels is at most lmbda. Large lambda -> less details
        :return:
        """
        cls_count = np.zeros(3)
        if lmbda <= self.nodes[self.current_sz - 1].alpha:
            # Set the output value of the root
            c = np.argmax(self.nodes[self.current_sz - 1].cls)
            self.nodes[self.current_sz - 1].label = cls_count[c]
            cls_count[c] += 1
            # Set colours of the other nodes
            for i in range(self.current_sz - 2, -1, -1):
                if self.is_level_root(i) and self.nodes[i].alpha >= lmbda:
                    # set the color of the level root
                    c = np.argmax(self.nodes[i].cls)
                    self.nodes[i].label = cls_count[c]
                    cls_count[c] += 1
                else:
                    # use parents' color
                    self.nodes[i].label = self.nodes[self.nodes[i].parent].label
        else:
            # set to black
            for i in range(0, self.current_sz):
                self.nodes[i].label = -1
        result = np.zeros((self.img.shape[0], self.img.shape[1]))
        cumsum = np.cumsum(cls_count)
        for i in range(self.img.shape[0] * self.img.shape[1]):
            result[i // self.img.shape[1], i % self.img.shape[1]] = self.nodes[i].label
            if np.argmax(self.nodes[i].cls) > 0:
                result[i // self.img.shape[1], i % self.img.shape[1]] += cumsum[np.argmax(self.nodes[i].cls)-1]

        return result, cls_count





    def get_cc_idxs(self, cc_type=None, label=None):
        assert (cc_type is not None) ^ (label is not None)
        node_idxs = []
        if label is None:
            if cc_type == CC_TYPE.FRGRND:
                for i in range(self.current_sz):
                    if self.is_level_root(i) and self.nodes[i].parent != -1 and  self.nodes[i].label >= 1:
                        node_idxs.append(i)

            elif cc_type==CC_TYPE.BCKGRND: # get background components
                for i in range(self.current_sz):
                    if self.is_level_root(i) and self.nodes[i].parent != -1 and self.nodes[i].label == 0:
                        node_idxs.append(i)
            else:
                for i in range(self.current_sz): # get all CCs regardless of a label
                    if self.is_level_root(i) and self.nodes[i].parent != -1:
                        node_idxs.append(i)
        else:
            for i in range(self.current_sz):
                if self.is_level_root(i) and self.nodes[i].label == label:
                    node_idxs.append(i)
        return  node_idxs

    def print_nodes(self, idx=None):
        if idx is None:
            idx = range(0, self.current_sz)
        for i in idx:
            #print("[ NODE #%d ] parent: %d, alpha: %d, label: %d" % (i, self.nodes[i].parent, self.nodes[i].alpha, self.nodes[i].label))
            print("[ NODE #%d ] " % (
            i))
            f = self.nodes[i].features
            print("area: ", self.nodes[i].area)
            print("alpha: %f" % self.nodes[i].alpha)
            print("Features: ", f)
            print(self.nodes[i].node_indices)
            print("========================")




