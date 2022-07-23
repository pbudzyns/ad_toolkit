import json
import pathlib
from typing import Dict, Any

import numpy as np
from nab.detectors.base import AnomalyDetector
from nab.runner import Runner

from sops_anomaly.detectors import RandomDetector, AutoEncoder


# def nab_evaluator(anomaly_detector: BaseModel):
#     class Detector(AnomalyDetector):
#
#         def handleRecord(self, inputData):
#             pass
#
#         def run(self):
#             data = np.array(self.dataSet.data['value'])
#             anomaly_detector.train(data)
#             anomalies = anomaly_detector.detect(data)
#
#             ans = self.dataSet.data.copy()
#             ans['anomaly_score'] = anomalies
#             return ans
#
#     return Detector

# class Detector(AnomalyDetector):
#
#     def __init__(self, *args, **kwargs):
#         self._ad = kwargs.pop('anomaly_detector')
#         super(Detector, self).__init__(*args, **kwargs)
#


class DetectorBase(AnomalyDetector):

    def handleRecord(self, inputData):
        pass

    def _run(self, anomaly_detector):
        data = np.array(self.dataSet.data['value'])
        anomaly_detector.train(data)
        anomalies = anomaly_detector.predict(data)

        ans = self.dataSet.data.copy()
        if len(anomalies) != len(ans['value']):
            anomalous_events = np.zeros_like(ans['value'])
            anomalous_events[anomalies] = 1
            ans['anomaly_score'] = anomalous_events
        else:
            ans['anomaly_score'] = anomalies
        return ans


class RandomNAB(DetectorBase):

    def run(self):
        detector = RandomDetector()
        return self._run(detector)


class AutoencoderNAB(DetectorBase):

    def run(self):
        detector = AutoEncoder(window_size=300, threshold=0.9)
        return self._run(detector)


def set_thresholds(detectors: Dict[str, Any], thresholds_file: pathlib.Path):
    template = {
        "reward_low_FN_rate": {
            "threshold": 0.5
        },
        "reward_low_FP_rate": {
            "threshold": 0.5
        },
        "standard": {
            "threshold": 0.5
        }
    }
    # with thresholds_file_path.open('rw') as thresholds_file:
    thr_file = json.loads(thresholds_file.read_text(encoding='utf-8'))
    for detector in detectors:
        if detector not in thr_file:
            thr_file[detector] = template
    thresholds_file.write_text(json.dumps(thr_file))
    # json.dump(thr_file, thresholds_file)


if __name__ == '__main__':

    root_dir = (
        pathlib.Path("..").absolute().parent.parent / "nab_files").absolute()
    data_dir = root_dir / "data"
    results_dir = root_dir / "results"
    windows_file = root_dir / "labels" / "combined_windows.json"
    profiles_file = root_dir / "config" / "profiles.json"
    thresholds_file = root_dir / "config" / "thresholds.json"

    runner = Runner(dataDir=str(data_dir),
                    labelPath=str(windows_file),
                    resultsDir=str(results_dir),
                    profilesPath=str(profiles_file),
                    thresholdPath=str(thresholds_file),
                    numCPUs=None)

    runner.initialize()

    detectors = {
        'random': RandomNAB,
        'ae': AutoencoderNAB,
        # 'betsi': BetsiNab,
    }
    # set_thresholds(detectors, thresholds_file)
    # runner.detect(detectors)

    # runner.optimize(list(detectors.keys()))

    with open(thresholds_file) as thresholdConfigFile:
        detectorThresholds = json.load(thresholdConfigFile)
    runner.score(list(detectors.keys()), detectorThresholds)

    # if args.normalize:
    #     try:
    runner.normalize()
