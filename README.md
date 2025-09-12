# Temporal_INR
Code for our papers "Temporal Periodic Image Registration with Implicit Neural Representations" and "Temporal Super-Resolution of Medical Images with Implicit Neural Representations" from the MLMI workshop at MICCAI 2025.

We utilize implicit neural representations to perform registration over periodic image data, here the lung CT from the [DIRLAB dataset](https://med.emory.edu/departments/radiation-oncology/research-laboratories/deformable-image-registration/index.html) and an in house heart CT dataset.

The network it self is trained for image registration over a periodic image sequence, but can also be used for creation of new images in the sequence.

## Usage



## Figures
General network architecture and illustration of point registration.

<img src="figures/method_fig.png" alt="Method overview" width="1000">

Results of temporal super resolution on 4D lung CT.

<img src="figures/lung_temporal_super_res.gif" alt="Temporal super resolution gif" width="1000">

## Citation
If you use this code, please cite our papers, depending on the usage

    @inproceedings{lowes2025TemporalINRReg,
        title={Temporal Periodic Image Registration with Implicit Neural Representations},
        author={Mathias Lowes and Kristine Aavild Sørensen and  Maxime Sermesant and Rasmus R. Paulsen},
        booktitle="Machine Learning in Medical Imaging",
        year="2025",
        publisher="Springer Nature Switzerland",
    }

    @inproceedings{lowes2025TemporalINRSuperRes,
        title={Temporal Super-Resolution of Medical Images with Implicit Neural Representations},
        author={Mathias Lowes and Kristine Aavild Sørensen and  Maxime Sermesant and Klaus Fuglsang Kofoed and Rasmus R. Paulsen},
        booktitle="Machine Learning in Medical Imaging",
        year="2025",
        publisher="Springer Nature Switzerland",
    }
