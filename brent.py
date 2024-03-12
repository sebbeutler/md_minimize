import numpy as np

# Définition de la fonction dont nous cherchons la racine
def fonction_objectif(x):
    return (x - 2) * (x + 3) * (x - 5)

# Implémentation de la méthode de Brent
def brent_method(fonction, a, b, tolérance=1e-6, max_iterations=100):
    # Initialisation des points et des valeurs de la fonction
    x1 = a
    x2 = b
    x3 = b
    x4 = 0
    x5 = 0
    fx1 = fonction(x1)
    fx2 = fonction(x2)

    # Vérification des signes initiaux de la fonction
    if np.sign(fx1) == np.sign(fx2):
        raise ValueError("Les valeurs de la fonction aux extrémités de l'intervalle ont la même signe.")

    fx3 = fx2
    
    # Boucle principale de l'algorithme de Brent
    for i in range(max_iterations):
        # Cas où les points x2 et x3 ont la même signe
        if (fx2 > 0.0 and fx3 > 0.0) or (fx2 < 0.0 and fx3 < 0.0):
            x4 = x1
            fx3 = fx1
            x5 = x2 - x1

        # Si la valeur absolue de fx3 est plus petite que celle de fx2, on ajuste les points
        if abs(fx3) < abs(fx2):
            x1 = x2
            x2 = x3
            x3 = x1
            fx1 = fx2
            fx2 = fx3
            fx3 = fx1

        # Calcul de la tolérance
        tol1 = 2.0 * np.finfo(float).eps * abs(x2) + 0.5 * tolérance
        m = 0.5 * (x3 - x2)

        # Si on atteint une tolérance acceptable ou si la fonction atteint 0, on retourne x2
        if abs(m) <= tol1 or fx2 == 0.0:
            return x2

        # Calcul des interpolations pour trouver un nouveau point
        if abs(x5) >= tol1 and abs(fx1) > abs(fx2):
            s = fx2 / fx1
            if x1 == x3:
                p = 2.0 * m * s
                q = 1.0 - s
            else:
                q = fx1 / fx3
                r = fx2 / fx3
                p = s * (2.0 * m * q * (q - r) - (x2 - x1) * (r - 1.0))
                q = (q - 1.0) * (r - 1.0) * (s - 1.0)

            if p > 0.0:
                q = -q

            p = abs(p)
            min1 = 3.0 * m * q - abs(tol1 * q)
            min2 = abs(x5 * q)
            if 2.0 * p < min(min1, min2):
                x5 = x4
                x4 = p / q
            else:
                x4 = m
                x5 = m

        else:
            x4 = m
            x5 = m

        # Mise à jour des points et des valeurs de la fonction
        x1 = x2
        fx1 = fx2
        if abs(x4) > tol1:
            x2 += x4
        else:
            x2 += np.sign(m) * tol1

        fx2 = fonction(x2)

    # Si la méthode ne converge pas, on lève une exception
    raise ValueError("La méthode de Brent n'a pas convergé après {} itérations.".format(max_iterations))

# Utilisation de la méthode de Brent pour trouver la racine de la fonction
racine = brent_method(fonction_objectif, -5, 5)

# Affichage du résultat
print("Résultat de la méthode de Brent :", racine)