"""
this file contains the definition of the Resnet architectures, please do take a look @
https://arxiv.org/pdf/1512.03385.pdf or https://towardsdatascience.com/an-overview-of-resnet-and-its-variants-5281e2f56035
for more details on these architectures.
Author: Kolade Gideon (Allaye) <no_email_yet>
"""
config = {
    "res18": [{
            "conv": [[3, 1, 0, 64, 64], [3, 1, 0, 64, 64]],
            "iteration": 2,
            "convolutional_block": False
        },
        {
            "conv": [[3, 1, 0, 64, 128], [3, 1, 0, 128, 128]],
            "iteration": 2,
            "convolutional_block": False
        },
        {
            "conv": [[3, 1, 0, 128, 256], [3, 1, 0, 256, 256]],
            "iteration": 2,
            "convolutional_block": False
        },
        {
            "conv": [[3, 1, 0, 256, 512], [3, 1, 0, 512, 512]],
            "iteration": 2,
            "convolutional_block": False
        }
    ],
    "res34": [{
        "conv": [[3, 64], [3, 64]],
        "iteration": 3,
        "convolutional_block": False
    },
        {
            "conv": [[3, 128], [3, 128]],
            "iteration": 4,
            "convolutional_block": False
        },
        {
            "conv": [[3, 256], [3, 256]],
            "iteration": 6,
            "convolutional_block": False
        },
        {
            "conv": [[3, 512], [3, 512]],
            "iteration": 3,
            "convolutional_block": False
        }
    ],
    "res50": [{
        "conv": [[1, 1, 0, 64, 64], [3, 1, 1, 64, 64], [1, 1, 0, 64, 256]],
        "iteration": 3,
        "convolutional_block": False
    },
        {
            "conv": [[1, 2, 1, 128], [3, 1, 1, 128], [1, 1, 1, 512]],
            "iteration": 4,
            "convolutional_block": True
        },
        {
            "conv": [[1, 256], [3, 256], [1, 1024]],
            "iteration": 6,
            "convolutional_block": True
        },
        {
            "conv": [[1, 512], [3, 512], [1, 2048]],
            "iteration": 3,
            "convolutional_block": True
        }
    ],
    "res101": [{
        "conv": [[1, 64], [3, 64], [1, 256]],
        "iteration": 3,
        "convolutional_block": False
    },
        {
            "conv": [[1, 128], [3, 128], [1, 512]],
            "iteration": 4,
            "convolutional_block": True
        },
        {
            "conv": [[1, 256], [3, 256], [1, 1024]],
            "iteration": 23,
            "convolutional_block": True
        },
        {
            "conv": [[1, 512], [3, 512], [1, 2048]],
            "iteration": 3,
            "convolutional_block": True
        }
    ],
    "res152": [{
        "conv": [[1, 64], [3, 64], [1, 256]],
        "iteration": 3,
        "convolutional_block": False
    },
        {
            "conv": [[1, 128], [3, 64], [1, 512]],
            "iteration": 8,
            "convolutional_block": True
        },
        {
            "conv": [[1, 256], [3, 256], [1, 1024]],
            "iteration": 36,
            "convolutional_block": True
        },
        {
            "conv": [[1, 512], [3, 512], [1, 2048]],
            "iteration": 3,
            "convolutional_block": True
        }
    ]
}
