import importlib.util, sys, types, torch

ROOT = "/Users/johannes/Repositories/schnetpack/src/schnetpack"

# Register lightweight stub packages so submodule imports resolve without
# executing schnetpack/__init__.py (which pulls in pytorch_lightning -> a
# broken torchvision in this env). We only need nn.* and representation.*.
for pkg, path in [
    ("schnetpack", ROOT),
    ("schnetpack.nn", f"{ROOT}/nn"),
    ("schnetpack.representation", f"{ROOT}/representation"),
]:
    m = types.ModuleType(pkg)
    m.__path__ = [path]
    sys.modules[pkg] = m

def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

# Load the pieces schnet.py needs, in dependency order, as real submodules.
import schnetpack
schnetpack.properties = _load("schnetpack.properties", f"{ROOT}/properties.py")
_load("schnetpack.nn.activations", f"{ROOT}/nn/activations.py")
_load("schnetpack.nn.base", f"{ROOT}/nn/base.py")
_load("schnetpack.nn.scatter", f"{ROOT}/nn/scatter.py")
# schnetpack.nn package namespace must expose Dense, scatter_add, shifted_softplus, replicate_module
import schnetpack.nn as _nn
from schnetpack.nn.base import Dense
from schnetpack.nn.scatter import scatter_add
from schnetpack.nn.activations import shifted_softplus
_nn.Dense = Dense; _nn.scatter_add = scatter_add; _nn.shifted_softplus = shifted_softplus
_load("schnetpack.nn.utils", f"{ROOT}/nn/utils.py")
from schnetpack.nn.utils import replicate_module
_nn.replicate_module = replicate_module
_load("schnetpack.representation.base", f"{ROOT}/representation/base.py")
_schnet = _load("schnetpack.representation.schnet", f"{ROOT}/representation/schnet.py")
_man = _load("schnetpack.representation.schnet_manual", f"{ROOT}/representation/schnet_manual.py")
SchNetInteraction = _schnet.SchNetInteraction
ManualSchNetInteraction = _man.ManualSchNetInteraction
interaction_fwd = _man.interaction_fwd
interaction_bwd = _man.interaction_bwd

torch.manual_seed(0)
B, R, F = 8, 5, 6        # n_atom_basis, n_rbf, n_filters
N, E = 7, 20             # atoms, edges

block = SchNetInteraction(n_atom_basis=B, n_rbf=R, n_filters=F).double()
man = ManualSchNetInteraction.from_torch(block)

idx_i = torch.randint(0, N, (E,))
idx_j = torch.randint(0, N, (E,))


def make_inputs(requires_grad):
    x = torch.randn(N, B, dtype=torch.double, requires_grad=requires_grad)
    f_ij = torch.randn(E, R, dtype=torch.double, requires_grad=requires_grad)
    rcut = torch.rand(E, dtype=torch.double, requires_grad=requires_grad)
    return x, f_ij, rcut


# ---- 1. forward parity: manual(x) == block(x) + x (block returns v only) ----
x, f_ij, rcut = make_inputs(False)
ref = block(x, f_ij, idx_i, idx_j, rcut) + x
out_fn = ManualSchNetInteraction.from_torch(block).forward(x, f_ij, idx_i, idx_j, rcut)
out_pure, saved = interaction_fwd(x, f_ij, idx_i, idx_j, rcut, man.params)
print("fwd vs ref (autograd path):", torch.allclose(out_fn, ref, atol=1e-10))
print("fwd vs ref (pure path)    :", torch.allclose(out_pure, ref, atol=1e-10))

# ---- 2. manual bwd vs autograd of the reference (x + v) ----
x, f_ij, rcut = make_inputs(True)
ref = block(x, f_ij, idx_i, idx_j, rcut) + x
g_out = torch.randn_like(ref)
gx_ref, gf_ref, grc_ref = torch.autograd.grad(ref, (x, f_ij, rcut), g_out)

_, saved = interaction_fwd(x.detach(), f_ij.detach(), idx_i, idx_j, rcut.detach(), man.params)
gx, gf, grc = interaction_bwd(g_out, saved, man.params)
print("bwd grad_x   :", torch.allclose(gx, gx_ref, atol=1e-9))
print("bwd grad_f_ij:", torch.allclose(gf, gf_ref, atol=1e-9))
print("bwd grad_rcut:", torch.allclose(grc, grc_ref, atol=1e-9))

# ---- 3. gradcheck the autograd.Function wrapper ----
x, f_ij, rcut = make_inputs(True)
ok = torch.autograd.gradcheck(
    lambda x, f, rc: man.forward(x, f, idx_i, idx_j, rc),
    (x, f_ij, rcut), atol=1e-6,
)
print("gradcheck InteractionFn:", ok)
