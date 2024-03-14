# InterpolacjaZadanie 1.Populacja Stanów Zjednoczonych na przestrzeni lat przedstawiałasię następująco:
# Rok   Populacja1900   76 212 1681910   92 228 4961920  106 021 5371930  123 202 6241940  132 164 5691950  151 325 7981960  179 323 1751970  203 302 0311980  226 542 199Istnieje dokładnie jeden wielomian ósmego stopnia, który interpoluje po-wyższe dziewięć punktów, natomiast sam wielomian może być reprezentowa-ny na różne sposoby. Rozważamy następujące zbiory funkcji bazowych
# φj(t),j= 1,...,9:φj(t) =tj−1(1)φj(t) = (t−1900)j−1(2)φj(t) = (t−1940)j−1(3)φj(t) = ((t−1940)/40)j−1(4)
# (a) Dla każdego z czterech zbiorów funkcji bazowych utwórz macierz Vander-monde’a.
# (b) Oblicz współczynnik uwarunkowania każdej z powyższch macierzy, używa-jąc funkcjinumpy.linalg.cond.
# (c) Używając najlepiej uwarunkowanej bazy wielomianów, znajdź współczyn-niki wielomianu interpolacyjnego dla danych z zadania. Narysuj wielomianinterpolacyjny. W tym celu użyj schematu Hornera i oblicz na przedziale[1900,1990] wartości wielomianu w odstępach jednorocznych. Na wykresieumieść także węzły interpolacji.1
# (d) Dokonaj ekstrapolacji wielomianu do roku 1990. Porównaj otrzymaną war-tość z prawdziwą wartością dla roku 1990, wynoszącą248 709 873. Ile wynosibłąd względny ekstrapolacji dla roku 1990?
# (e) Wyznacz wielomian interpolacyjny Lagrange’a na podstawie 9 węzłów in-terpolacji podanych w zadaniu. Oblicz wartości wielomianu w odstępachjednorocznych.
# (f) Wyznacz wielomian interpolacyjny Newtona na podstawie tych samych wę-złów interpolacji i oblicz wartości wielomianu w odstępach jednorocznych.
# (g) Zaokrąglij dane podane w tabeli do jednego miliona. Na podstawie takichdanych wyznacz wielomian interpolacyjny ósmego stopnia, używając naj-lepiej uwarunkowanej bazy z podpunktu 
# (c). Porównaj wyznaczone współ-czynniki z współczynnikami obliczonymi w podpunkcie 
# (c). Wyjaśnij otrzy-many wynik.2