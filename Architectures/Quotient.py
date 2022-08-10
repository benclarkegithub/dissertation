from torch import nn


class Quotient(nn.Module):
    def __init__(self,
                 canvas,
                 encoder,
                 lats_to_dec,
                 decoder,
                 *,
                 KL=None,
                 enc_to_lat=None,
                 enc_to_lats=None):
        super().__init__()

        # Canvas
        self.canvas = canvas
        # KL
        self.KL = KL
        # Encoder
        self.encoder = encoder
        # Encoder to Latents
        if enc_to_lat:
            self.enc_to_lat = enc_to_lat
        if enc_to_lats:
            self.enc_to_lats = nn.ModuleList([x for x in enc_to_lats])
        # Latents to Decoder
        self.lats_to_dec = nn.ModuleList([x for x in lats_to_dec])
        # Decoder
        self.decoder = decoder
