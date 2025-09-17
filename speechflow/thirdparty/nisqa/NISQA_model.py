"""
@author: Gabriel Mittag, TU-Berlin
"""
import os
import typing as tp

from pathlib import Path

import torch
import pandas as pd

from speechflow.thirdparty.nisqa import NISQA_lib as NL
from speechflow.utils.fs import get_root_dir


class nisqaModel:
    """
    nisqaModel: Main class that loads the model and the datasets. Contains
    the training loop, prediction, and evaluation function.
    """

    def __init__(self, device: str):
        self._device = torch.device(device)
        args = {
            "mode": "predict_file",
            "model": "NISQA_DIM",
            "pretrained_model": get_root_dir() / "speechflow/data/nisqa/nisqa.tar",
            "ms_channel": 1,
        }
        self._model, self._args = self._loadModel(args)
        self._model.eval().to(self._device)

    def predict(self, file_path: Path) -> tp.Dict[str, float]:
        ds_val = self._loadDatasetsFile(file_path, self._args)
        return NL.predict_dim(self._model, ds_val, self._device)

    def _loadDatasetsFile(self, file_path: Path, args):
        df_val = pd.DataFrame([file_path.name], columns=["deg"])
        ds_val = NL.SpeechQualityDataset(
            df_val,
            df_con=None,
            data_dir=file_path.parent.as_posix(),
            filename_column="deg",
            mos_column="predict_only",
            seg_length=args["ms_seg_length"],
            max_length=args["ms_max_segments"],
            to_memory=None,
            to_memory_workers=None,
            seg_hop_length=args["ms_seg_hop_length"],
            transform=None,
            ms_n_fft=args["ms_n_fft"],
            ms_hop_length=args["ms_hop_length"],
            ms_win_length=args["ms_win_length"],
            ms_n_mels=args["ms_n_mels"],
            ms_sr=args["ms_sr"],
            ms_fmax=args["ms_fmax"],
            ms_channel=args["ms_channel"],
            double_ended=args["double_ended"],
            dim=args["dim"],
            filename_column_ref=None,
        )
        return ds_val

    def _loadModel(self, args):
        """Loads the Pytorch models with given input arguments."""
        # if True overwrite input arguments from pretrained model
        if args["pretrained_model"]:
            if os.path.isabs(args["pretrained_model"]):
                model_path = os.path.join(args["pretrained_model"])
            else:
                model_path = os.path.join(os.getcwd(), args["pretrained_model"])
            checkpoint = torch.load(model_path, map_location="cpu")

            # update checkpoint arguments with new arguments
            checkpoint["args"].update(args)
            args = checkpoint["args"]

        if args["model"] == "NISQA_DIM":
            args["dim"] = True
            args["csv_mos_train"] = None  # column names hardcoded for dim models
            args["csv_mos_val"] = None
        else:
            args["dim"] = False

        if args["model"] == "NISQA_DE":
            args["double_ended"] = True
        else:
            args["double_ended"] = False
            args["csv_ref"] = None

        # Load Model
        model_args = {
            "ms_seg_length": args["ms_seg_length"],
            "ms_n_mels": args["ms_n_mels"],
            "cnn_model": args["cnn_model"],
            "cnn_c_out_1": args["cnn_c_out_1"],
            "cnn_c_out_2": args["cnn_c_out_2"],
            "cnn_c_out_3": args["cnn_c_out_3"],
            "cnn_kernel_size": args["cnn_kernel_size"],
            "cnn_dropout": args["cnn_dropout"],
            "cnn_pool_1": args["cnn_pool_1"],
            "cnn_pool_2": args["cnn_pool_2"],
            "cnn_pool_3": args["cnn_pool_3"],
            "cnn_fc_out_h": args["cnn_fc_out_h"],
            "td": args["td"],
            "td_sa_d_model": args["td_sa_d_model"],
            "td_sa_nhead": args["td_sa_nhead"],
            "td_sa_pos_enc": args["td_sa_pos_enc"],
            "td_sa_num_layers": args["td_sa_num_layers"],
            "td_sa_h": args["td_sa_h"],
            "td_sa_dropout": args["td_sa_dropout"],
            "td_lstm_h": args["td_lstm_h"],
            "td_lstm_num_layers": args["td_lstm_num_layers"],
            "td_lstm_dropout": args["td_lstm_dropout"],
            "td_lstm_bidirectional": args["td_lstm_bidirectional"],
            "td_2": args["td_2"],
            "td_2_sa_d_model": args["td_2_sa_d_model"],
            "td_2_sa_nhead": args["td_2_sa_nhead"],
            "td_2_sa_pos_enc": args["td_2_sa_pos_enc"],
            "td_2_sa_num_layers": args["td_2_sa_num_layers"],
            "td_2_sa_h": args["td_2_sa_h"],
            "td_2_sa_dropout": args["td_2_sa_dropout"],
            "td_2_lstm_h": args["td_2_lstm_h"],
            "td_2_lstm_num_layers": args["td_2_lstm_num_layers"],
            "td_2_lstm_dropout": args["td_2_lstm_dropout"],
            "td_2_lstm_bidirectional": args["td_2_lstm_bidirectional"],
            "pool": args["pool"],
            "pool_att_h": args["pool_att_h"],
            "pool_att_dropout": args["pool_att_dropout"],
        }

        if args["double_ended"]:
            model_args.update(
                {
                    "de_align": args["de_align"],
                    "de_align_apply": args["de_align_apply"],
                    "de_fuse_dim": args["de_fuse_dim"],
                    "de_fuse": args["de_fuse"],
                }
            )

        # print('Model architecture: ' + args['model'])
        if args["model"] == "NISQA":
            model = NL.NISQA(**model_args)
        elif args["model"] == "NISQA_DIM":
            model = NL.NISQA_DIM(**model_args)
        elif args["model"] == "NISQA_DE":
            model = NL.NISQA_DE(**model_args)
        else:
            raise NotImplementedError("Model not available")

        # Load weights if pretrained model is used
        if args["pretrained_model"]:
            missing_keys, unexpected_keys = model.load_state_dict(
                checkpoint["model_state_dict"], strict=True
            )
            if missing_keys:
                print("missing_keys:")
                print(missing_keys)
            if unexpected_keys:
                print("unexpected_keys:")
                print(unexpected_keys)

        return model, args
