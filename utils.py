import unicodedata
import re
import torch


# Turn a Unicode string to plain ASCII, thanks to
# http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s


def criterion(inp, out, mask):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ceLoss = - torch.log(torch.gather(inp, 1, out.view(-1, 1)))
    return ceLoss.masked_select(mask).mean().to(device)


