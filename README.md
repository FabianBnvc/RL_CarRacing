## Vorraussetzungen 

Für dieses Projekt wird Visual Studio Community 2022 zumsammmen mit der Entwicklererweiterung von C++ benötigt. 

Die genutzte Pythonversion ist die 3.10.0

Alle Python Packete können über die requirements.txt datei über "pip install -r requirements.txt" runtergeladen werden. 


### Runs

Die einzelnen Durchläufe (Runs) werden anhand eines Zeitstempels erstellt. Das bedeutet, dass der älteste Durchlauf oben und der neueste unten zu finden ist. Jeder Durchlauf enthält eine Abbildung, die den Verlauf von Verlust (Loss) und Belohnung (Reward) während des Trainings darstellt, sowie die dazugehörigen CSV-Dateien. Das trainierte Modell wird ebenfalls gespeichert. Basierend auf den Ergebnissen wurden für einige Durchläufe auch Visualisierungen erstellt. Bei schlechten Ergebnissen sind daher möglicherweise keine Visualisierungen/Testruns vorhanden.


### Quelle

https://github.com/andywu0913/OpenAI-GYM-CarRacing-DQN/tree/master