models = {
    # Mine
    'SJM-2022': [['Conv1D', [40, 11, 0]],
           ['Conv1D', [80, 22, 0]],
           ['Conv1D', [120, 33, 0]],
           ['Conv1D', [160, 44, 0]],
           ['Conv1D', [200, 55, 0]],
           ['Flatten'],
           ['Dense', [128, 0.0]],
           ['Dense', [32, 0.0]],
           ['Final']
           ],

    'test':
        [['Conv1D', [32, 7, 0, 'valid', 'relu']],
         ['Conv1D', [32, 7, 0, 'valid', 'relu']],
         ['Conv1D', [64, 5, 0, 'valid', 'relu']],
         ['Conv1D', [64, 3, 0, 'valid', 'relu']],
         ['Conv1D', [128, 3, 0, 'valid', 'relu']],
         ['Conv1D', [128, 3, 0, 'valid', 'relu']],
         ['Conv1D', [256, 3, 0, 'valid', 'relu']],
         ['Conv1D', [256, 3, 0, 'valid', 'relu']],
         ['Flatten'],
         ['Dense', [128, 0.0, 'relu']],
         ['Dense', [32, 0.0, 'relu']],
         ['Final']
         ],

    'mini':
        [['Flatten'],
         ['Dense', [4, 0.0, 'relu']],
         ['Final']
         ],

    'SJM_12_2023':
        [['Conv1D', [32, 7, 0, 'valid', 'relu']],
         ['Conv1D', [32, 7, 0, 'valid', 'relu']],
         ['Conv1D', [64, 5, 0, 'valid', 'relu']],
         ['Conv1D', [64, 3, 0, 'valid', 'relu']],
         ['Conv1D', [128, 3, 0, 'valid', 'relu']],
         ['Conv1D', [128, 3, 0, 'valid', 'relu']],
         ['Conv1D', [256, 3, 0, 'valid', 'relu']],
         ['Conv1D', [256, 3, 0, 'valid', 'relu']],
         ['Flatten'],
         ['Dense', [128, 0.0, 'relu']],
         ['Dense', [32, 0.0, 'relu']],
         ['Final']
         ],

    'VAE':
        [['Conv1D', [32, 7, 0, 'valid', 'relu']],
         ['Conv1D', [32, 7, 0, 'valid', 'relu']],
         ['Conv1D', [64, 5, 0, 'valid', 'relu']],
         ['Conv1D', [64, 3, 0, 'valid', 'relu']],
         ['Conv1D', [128, 3, 0, 'valid', 'relu']],
         ['Conv1D', [128, 3, 0, 'valid', 'relu']],
         ['Conv1D', [256, 3, 0, 'valid', 'relu']],
         ['Conv1D', [256, 3, 0, 'valid', 'relu']],
         ],

    'ResNet': ['ResNet', 3*9+2],

    'Inception': ['Inception', None],

    # from Attia Z, Nature 2019
    'attia1': [[16, 5], [16, 5], [32, 5], [32, 3], [64, 3], [64, 3]],

    # from Attia Z, Circulation 2019
    'attia2': [[16, [7]], [16, [5]], [32, [5]], [32, [5]], [64, [5]], [64, [3]], [64, [3]], [64, [3]]],

    # from Qihang Yao, Information Fusion 2020
    'yao': [['Conv1D', [64, 3, 0]],
            ['Conv1D', [64, 3, 0]],
            ['MaxPool', [3, 3]],
            ['Conv1D', [128, 3, 0]],
            ['Conv1D', [128, 3, 0]],
            ['MaxPool', [3, 3]],
            ['Conv1D', [256, 3, 0]],
            ['Conv1D', [256, 3, 0]],
            ['Conv1D', [256, 3, 0]],
            ['MaxPool', [3, 3]],
            ['Conv1D', [256, 3, 0]],
            ['Conv1D', [256, 3, 0]],
            ['Conv1D', [256, 3, 0]],
            ['MaxPool', [3, 3]],
            ['Conv1D', [256, 3, 0]],
            ['Conv1D', [256, 3, 0]],
            ['Conv1D', [256, 3, 0]],
            ['MaxPool', [3, 3]],
            ['Flatten'],
            ['Dense', [64, 0.05]],
            ['Final']
            ]
}



