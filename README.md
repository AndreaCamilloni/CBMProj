# Skin cancer diagnosis using pre-trained neural networks in the concept-bottleneck models framework

Deep learning techniques have been widely used in the medical field for image
classification and in the past few years have shown to be successful in providing good
diagnostic accuracy.
Skin cancer is a common and deadly disease that a Deep Convolutional Neural Network(CNN) could detect. Clinical images and images acquired by using a particular
handheld instrument , called Dermatoscope, could be used in order to provide a
diagnosis. To study this possibility, a few datasets have been released over the years.
This work focuses on the Derm7pt dataset, which provides labels for 7 clinically
significant attributes in addition to the diagnosis of the lesion. These attributes
are part of the 7-point checklist method used in clinical practice. The goal of this
thesis is to provide a model able to detect Melanoma in dermoscopic images of skin
lesions, by learning first a set of human-understandable concepts. To do this, the
Concept Bottleneck Framework was employed, by designing models that first predict
the clinical attributes from 7-point checklist method and then use those for the
final diagnosis of the lesion. These models have then been compared with the more
widespread Single Task Learning approaches, which learn the diagnosis end-to-end
without intermediate concepts.
The bottleneck model that showed the best performance in melanoma diagnosis
achieved an accuracy and F1 score of 77.50%, and 65.60% respectively. The Single
Task Learning approach obtained the best result by achieving an accuracy and F1
score of 81.25%, and 67.96%.
The experiments have shown that the performance of the black-box models is slightly
superior. However, they lose the ability to provide further insights into prediction as
Concepts Bottleneck Models can do.

# Diagnosi di tumori cutanei tramite reti neurali pre-addestrate nel framework concept-bottleneck models 
Le tecniche di deep learning sono state ampiamente utilizzate in campo medico
per la classificazione delle immagini e negli ultimi anni hanno dimostrato di essere
efficaci nel fornire una buona accuratezza diagnostica.
Il tumore della pelle, identificato in alcune forme come Melanoma, è una causa
comune di morte nella popolazione odierna, se non diagnosticato precocemente;
le reti neurali convoluzionali profonde, o CNN, possono rilevarlo a partire da una
semplice immagine. Immagini acquisite clinicamente, o tramite appositi strumenti,
come ad esempio il Dermatoscopio, il quale mette in evidenza pattern non visibili ad
occhio nudo di una lesione cutanea, possono esser analizzate da una CNN per fornire
una diagnosi precoce.
Negli anni, alcuni dataset sono stati resi disponibili per dare la possibilità di approfondire la materia.
In questo lavoro, è stato usato il dataset Derm7pt, il quale fornisce le labels per 7
importanti attributi clinici, in aggiunta alla diagnosi della relativa lesione cutanea.
Questi attributi sono la caratteristica principale del metodo di classificazione, usato
in ambiente clinico dai dermatologi, conosciuto come 7-point checklist. Questa tesi
propone diversi metodi per classificare una lesione cutanea, differenziandola tra
Melanoma e Nevi.
Un grande ostacolo incontrato nei metodi proposti nella letteratura è quello rappresentato dalla natura delle reti neurali, le quali sono delle “scatole chiuse”, che non
forniscono un’ interpretazione umana di come hanno effettuato le loro predizioni.
Quindi un particolare focus, nel lavoro proposto, è andato ad architetture conosciute
come Concept Bottleneck Models(CBM), le quali apprendono un set di concetti
intermedi (i 7 attributi del 7-pt Checklist), interpretabili dall’umano e fanno infine
la predizione sulla diagnosi.
Sono state poi confrontate le diverse architetture basate sul framework dei Concept
Bottleneck Models con architetture end-to-end, cioè delle scatole chiuse che predicono
direttamente la diagnosi.
Gli esperimenti hanno mostrato i migliori risultati sulla diagnosi finale, ottenuti
dai modelli end-to-end, seppur perdendo l’abilità di fornire una spiegazione della
predizione, come i CBM possono fare.
In particolare, la miglior accuratezza nella predizione è stata fornita dal modello
costituito da una ResidualNet pre-addestrata, i cui layer finali sono stati addestrati
nuovamente su un test set del dataset proposto, ottenendo circa un’accuratezza del
81.25%, e un F1 score del 67.96%.
Mentre l’implementazione nel framework CBM, che ha ottenuto la miglior performance è stata l’archittettura Sequential, nella quale addestrando prima la base del
modello costituita da un InceptionNet per predire i concetti e, allenando successivamente la testa del modello con i concetti predetti, ha ottenuto rispettivamente
un’accuratezza e un F1 score del 77.50% e 65.60%.
Nel complesso le architetture proposte hanno comunque ottenuto buoni risultati,
comparabili con le regole, come ad esempio la 7-point Checklist Rule, proposte nella
letteratura. Applicando infatti la precedente regola sui concetti reali, sono stati
ottenuti i seguenti risultati 83.44% e 78.43% rispettivamente per le 2 metriche.
