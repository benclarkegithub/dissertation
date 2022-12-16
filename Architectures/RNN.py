from torch import nn


class RNN(nn.Module):
    def __init__(self,
                 canvas,
                 encoder,
                 lats_to_dec,
                 decoder,
                 *,
                 encoder_2=None,
                 enc_enc_to_enc=None,
                 enc_to_lat=None,
                 enc_to_lats=None,
                 enc_lats_to_lats=None,
                 enc_enc_to_lats=None,
                 lats_to_lats=None):
        super().__init__()

        # Canvas
        self.canvas = canvas
        # Encoder
        self.encoder = encoder
        # Encoder 2
        if encoder_2:
            self.encoder_2 = encoder_2
        # Encoder Encoder to Encoder
        if enc_enc_to_enc:
            self.enc_enc_to_enc = enc_enc_to_enc
        # Encoder to Latents
        if enc_to_lat:
            self.enc_to_lat = enc_to_lat
        if enc_to_lats:
            self.enc_to_lats = nn.ModuleList([x for x in enc_to_lats])
        # Encoder Latents to Latents
        if enc_lats_to_lats:
            self.enc_lats_to_lats = nn.ModuleList([x for x in enc_lats_to_lats])
        # Encoder Encoder to Latents
        if enc_enc_to_lats:
            self.enc_enc_to_lats = nn.ModuleList([x for x in enc_enc_to_lats])
        # Latents to Latents
        if lats_to_lats:
            self.lats_to_lats = lats_to_lats
        # Latents to Decoder
        self.lats_to_dec = nn.ModuleList([x for x in lats_to_dec])
        # Decoder
        self.decoder = decoder
