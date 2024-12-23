# heart-rate-variability-analysis
This repository contains all the source code files for the data science project Investigating Heart Rate Variability

**Background Information**
The human heart consists of a series of chambers surrounded by muscles that contract and relax rhythmically to pump blood around the human body.

![image](https://github.com/user-attachments/assets/c6eb902a-3641-48a3-8b41-079d5be6bb93)

The behaviour and health of the heart can be studied using electro-cardiogram or ECG recordings, in which electrodes capture the electrical activity associated with the beating heart.

The electrical impulses to trigger heartbeats are generated by specialised pacemaker cells in the cardiac muscle. Normal or “sinus” rhythm heartbeats are triggered when pacemaker cells located in the sinoatrial node depolarise causing an impulse that propagates as a wave, triggering the pacemaker cells of the other regions in a coordinated sequence, such that a characteristic waveform:

![image](https://github.com/user-attachments/assets/454c95ca-2ecb-4fb7-88ef-62602277ce6d)

However it is also possible for pacemaker cells of the atrial or ventricular regions to spontaneously depolarise without receiving a signal from the sinoatrial node. This triggers what is known as a premature or ectopic heart beat. These are classified by where the premature depolarisation occurs:

- PAC Premature atrial contraction
- PVC Premature ventricular contraction

Such contractions may occur at a low rate in healthy individuals, but may be more frequently occurring or occur with characteristic repetitions in individuals with heart conditions.

**Analysis of ECG recordings and Heart Rate Variability**

To assess heart health a patient is asked to wear an ECG monitor that records their hearts electrical activity over an extended period. The recording is then annotated to:
- identify each beat by referencing the location of the R wave of the QRS complex.
- labels the type of each beat using N for normal sinus rhythm or another character to signify the type of non-normal type.
The resulting ECG can then analysed by an expert to look at, so they can try to diagnose the patient’s heart health.
In addition to a manual inspection, several metrics can be automatically calculated from the annotated ECG recording. The most basic metric is the average interval beat interval (e.g. in ms) which may also be written as average beats-per-minute (BPM) value.

**mean BPM = (1000 / mean beat interval in 𝐦𝐬)× 60**

When calculated from the full set of annotated beats, this is known as the average R-R interval. However to better characterise the interval of the heart in its normal sinus rhythm,  another reported metric might be the average N-N interval, which is found by filtering the recording so that only the intervals between consecutive N-type beats are included. The rate at which hearts beat reflects the needs of the circulatory system to supply the body with oxygenated blood, such that the heart rate will be higher during physical exercise and lower during rest. However, in addition to this type of variability which reflects the changing needs of the body, the heart rate is also observed to vary beat-by-beat.

Studies have shown that poor heart health is associated with decreased beat-to-beat variability (where beats occur in a regular frequency like a metronome). Therefore there has been considerable effort to come up with ways to measure the variability of the heart and explore more fully how it varies with characteristics such as
gender, age and heart health. In this report we will study the following metrics which can be calculated from an annotated ECG recording. These include calculations based on the difference in beat-to-beat intervals, for example, if a beat of 800ms duration is followed by a beat of 870ms duration, then there is a beat-to-beat variation of +70ms. 
