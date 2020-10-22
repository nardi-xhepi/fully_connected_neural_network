PROBABILITIES = 0
VALUE = 1

import numpy as np
from NeuralNetwork import Network as net


possible_actions = {}

s = 0
for i in range(1, 17):
    for j in range(1, 17):
        for k in range(3):
            s += 1
            possible_actions[s] = (i, j, k)

def _get_possible_actions_for_actual_state(etatDuJeu):
    """
    Accès à la base de données pour obtenir l'état du jeu et les actions possible à jouer.
    On renvoie une liste de taille 300 (la taille de toutes les actions possibles avec des 0 aux indices qui correspondent à une action qu'on ne peut pas prendre et des 1 là où on peut prendre une action

    """
    pass


def _get_game_state(jeu):
    pass

def _is_game_finished():
    """
    Verifier si le jeu est terminé

    """
    pass

def _winner():
    """
    Renvoyer 1 si j'ai gagné, -1 sinon

    """
    pass


class Noeud:
    def __init__(self, state, parent, action):
        self.state = state
        self.parent = parent
        self.action = action
        self.Ns = 0
        self.edges = {}

    def est_feuille(self):
        return len(self.edges) == 0

    def ajouter_enfant(self, edge_action, P):
        self.edges[edge_action] = {
                                    "N_s_a": 0,
                                    "W_s_a": 0,
                                    "Q_s_a": 0,
                                    "P_s_a": P
                                }


class MCTS:
    def __init__(self, state, network, jeu, cupt = 1, temperature = 1):
        self.root = Noeud(state, None, None)
        self.arbre = {}
        self.network = network
        self.cupt = cupt
        self.jeu = jeu
        self.temperature = temperature

    def get_action_probability(self, state):
        """
        Renvoyer le vecteur contenant les probabilités pour chaque action selon l'état

        """

        vecteur_probabilite = [0 for _ in range(len(possible_actions))]
        noeud = self.arbre[state]
        for (action, c_a) in noeud.edges.items():
            index = list(possible_actions.keys())[list(possible_actions.values()).index(action)]
            vecteur_probabilite[index - 1] = c_a["N_s_a"] / noeud.Ns
        return vecteur_probabilite

    def _expand_noeud(self, noeud):
        actions = _get_possible_actions_for_actual_state(state)
        prediction = network.predict(state)
        Ps, v = prediction[PROBABILITIES], prediction[VALUE]
        for i in range(len(actions)):
            if actions[i]:
                noeud.edges[possible_actions[i + 1]] = {    "N_s_a": 0,
                                                            "W_s_a": 0,
                                                            "Q_s_a": 0,
                                                            "P_s_a": Ps[0][i]
                                                    }
        return v

    def search_and_backup(self, st):
        state = st
        if self.root.est_feuille():
            v = self._expand_noeud(self.root)

        noeud = self.root
        max_u = -float("inf")
        best_action = -1

        while not noeud.est_feuille():
            for (action, characteristics) in noeud.edges.items():
                P_a = characteristics["P_s_a"]; Q_a = characteristics["Q_s_a"]; N_a = characteristics["N_s_a"]; N = noeud.Ns

                U = self.cupt * P_a * ( N** 0.5 ) / N_a

                if Q + U > max_u:
                    max_u = Q + U
                    best_action = action

            older_state = state
            state = self.jeu(best_action)
            if (older_state, best_action) in self.arbre:
                noeud = self.arbre[(older_state, best_action)]
            else:
                noeud = Noeud(state, noeud, best_action)
                self.arbre[(older_state, best_action)] = noeud

        game_finished = _is_game_finished()
        if noeud.est_feuille() and not game_finished:
            v = self._expand_noeud(noeud)

        if game_finished:
            v = _winner()

        while noeud != None:
            previous_action = noeud.action
            noeud = noeud.parent
            c_a = noeud.edges[previous_action]
            c_a["N_s_a"] += 1
            c_a["W_s_a"] += v
            c_a["Q_s_a"] +=  c_a["W_s_a"]/c_a["N_s_a"]
            noeud.Ns += c_a["N_s_a"] ** self.temperature


class NeuralNetworks:
    def __init__(self, nr_entrees, nr_d_actions):
        self.reseau_probas = net(optimizer = "Adam")
        self.reseau_values = net(optimizer = "Adam")

        self.reseau_probas.add_layer(nr_entrees, 64, "sigmoid")
        self.reseau_probas.add_layer(64, 32, "sigmoid")
        self.reseau_probas.add_layer(32, 64, "sigmoid")
        self.reseau_probas.add_layer(64, 64, "sigmoid")
        self.reseau_probas.add_layer(64, 64, "sigmoid")
        self.reseau_probas.add_layer(64, 128, "sigmoid")
        self.reseau_probas.add_layer(128, nr_d_actions, "sigmoid")

        self.reseau_values.add_layer(nr_entrees, 64, "sigmoid")
        self.reseau_values.add_layer(64, 32, "sigmoid")
        self.reseau_values.add_layer(32, 1, "tanh")

    def predict(self, entree):
        vecteur_probas = self.reseau_probas.predict(entree)
        valeur_de_gain = self.reseau_values.predict(entree)
        return vecteur_probas, valeur_de_gain

    def train(self, donnees):
        X, Y_P, Y_V = donnees
        for _ in range(len(X_P)):
            self.reseau_probas.train(X[i], Y_P[i], 0.001)
            self.reseau_values.train(X[i], Y_V[i], 0.001)



def _jouer_match(old_network, new_network):
    """ renvoyer 1 si new_network gagne, 0 sinon """
    pass


def _evaluate_network(old_network, new_network, number_of_plays):

    """
    Jouer plusieurs parties de jeu entre les deux réseaux et choisir celui qui gange plus que 55% des matchs
    """
    s = 0
    for _ in range(number_of_plays):
        s += _jouer_match(old_network, new_network)
    if s / number_of_plays >= 0.55:
        return new_network
    else:
        return old_network
