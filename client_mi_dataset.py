import json
import os

import datasets

_DESCRIPTION = """\
AnnoMI dataset for clients: https://www.mdpi.com/1999-5903/15/3/110
"""

_HOMEPAGE = "https://github.com/uccollab/AnnoMI"

ANNOMI_DATA_DIR = './data/client_splits_with_certainty'


class AnnoMIDataset(datasets.GeneratorBasedBuilder):
    """ AnnoMI Dataset which consists only of client utterances. """

    VERSION = datasets.Version("3.2.0")  # add multitask output

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="attitude-full", version=VERSION,
                               description="Use all training samples for in-context settings."),
        datasets.BuilderConfig(name="attitude-partial", version=VERSION,
                               description="Use the most diverse 300 training samples for in-context settings."),
        datasets.BuilderConfig(name="certainty", version=VERSION,
                               description="Use dataset with strength annotation."),
        datasets.BuilderConfig(name="multitask-certainty", version=VERSION,
                               description="Use dataset with strength annotation."),
        datasets.BuilderConfig(name="multitask-full", version=VERSION,
                               description="Use dataset with strength annotation mixed with all attitude samples."),
        datasets.BuilderConfig(name="multitask-300", version=VERSION,
                               description="Use dataset with strength annotation mixed with 300 attitude samples."),
        datasets.BuilderConfig(name="multitask-200", version=VERSION,
                               description="Use dataset with strength annotation mixed with 200 attitude samples."),
        datasets.BuilderConfig(name="multitask-100", version=VERSION,
                               description="Use dataset with strength annotation mixed with 100 attitude samples."),
    ]

    DEFAULT_CONFIG_NAME = "partial"

    def _info(self):
        # This method specifies the datasets.DatasetInfo object which contains information and typings for the dataset
        if "attitude" in self.config.name:
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "topic": datasets.Value("string"),
                    "video_url": datasets.Value("string"),
                    "timestamp": datasets.Value("string"),
                    "quality": datasets.Value("string"),
                    "client_utterance": datasets.Value("string"),
                    "prev_therapist_utt": datasets.Value("string"),
                    "target": datasets.Value("string"),
                }
            )
        else:
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "client_utterance": datasets.Value("string"),
                    "prev_therapist_utt": datasets.Value("string"),
                    "target": datasets.Value("string"),
                }
            )
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,  # Here we define them above because they are different between the two configurations
            # If there's a common (input, target) tuple from the features, uncomment supervised_keys line below and
            # specify them. They'll be used if as_supervised=True in builder.as_dataset.
            # supervised_keys=("sentence", "label"),
            homepage=_HOMEPAGE,
        )

    def _split_generators(self, dl_manager):
        test_dir = ANNOMI_DATA_DIR
        data_dir = ANNOMI_DATA_DIR

        train_file, val_file = "", ""
        if "attitude" in self.config.name:
            data_dir = f"{ANNOMI_DATA_DIR}/attitude"
            val_file = f"validation.json"
            train_file = f"train_full.json" if "full" in self.config.name else f"train_partial.json"

        elif "certainty" in self.config.name:  # and "multitask" not in self.config.name
            data_dir = f"{ANNOMI_DATA_DIR}/certainty"
            train_file = f"train.json"
            val_file = f"validation.json"

        elif "multitask" in self.config.name:
            data_dir = f"{ANNOMI_DATA_DIR}/multitask"
            val_file = f"validation.json"
            if "full" in self.config.name:
                train_file = f"train_full.json"
            elif "300" in self.config.name:
                train_file = f"train_mixed300.json"
            elif "200" in self.config.name:
                train_file = f"train_mixed200.json"
            elif "100" in self.config.name:
                train_file = f"train_mixed100.json"

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, train_file),
                    "split": "train"
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, val_file),
                    "split": "validation"
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(test_dir, f"test.json"),
                    "split": "test"
                },
            ),
        ]

    def _generate_examples(self, filepath, split):
        # TThis method handles input defined in _split_generators to yield (key, example) tuples from the dataset.
        # The `key` is for legacy reasons (tfds) and is not important in itself, but must be unique for each example.

        with open(filepath, 'r', encoding='utf-8') as file:
            data = json.load(file)

        if "attitude" in self.config.name:
            for utt_id, utt_info in data.items():
                yield utt_id, {
                    "id": utt_id,
                    "topic": utt_info['meta_info']['topic'],
                    "video_url": utt_info['meta_info']['video_url'],
                    "timestamp": utt_info['meta_info']['timestamp'],
                    "quality": utt_info['meta_info']['quality'],
                    "client_utterance": utt_info['utt_info']['text'],
                    "prev_therapist_utt": utt_info['utt_info']['prev_utt'],
                    "target": utt_info['utt_info']['behaviour_codes']['main_type'][0],
                }

        elif "certainty" in self.config.name and "multitask" not in self.config.name:
            for utt_id, utt_info in data.items():
                yield utt_id, {
                    "id": utt_id,
                    "client_utterance": utt_info['utt_info']['text'],
                    "prev_therapist_utt": utt_info['utt_info']['prev_utt'],
                    "target": utt_info['utt_info']['certainty'],
                }

        elif "multitask" in self.config.name:

            # output_mapping = {
            #     "change high": "(a) change high",
            #     "change medium": "(b) change medium",
            #     "change low": "(c) change low",
            #     "neutral high": "(d) neutral high",
            #     "neutral medium": "(e) neutral medium",
            #     "neutral low": "(f) neutral low",
            #     "sustain high": "(g) sustain high",
            #     "sustain medium": "(h) sustain medium",
            #     "sustain low": "(i) sustain low"
            # }

            for utt_id, utt_info in data.items():
                target = f"{' '.join(utt_info['utt_info']['behaviour_codes']['main_type'])} " \
                         f"{utt_info['utt_info']['certainty']}".strip()

                yield utt_id, {
                    "id": utt_id,
                    "client_utterance": utt_info['utt_info']['text'],
                    "prev_therapist_utt": utt_info['utt_info']['prev_utt'],
                    "target": target,
                }
