"""Patch the reference checkouts in place, idempotently.

We keep FragFM and ms-pred as pristine upstream clones and repair them here, so
the exact delta we depend on stays visible and drops out the moment upstream
fixes it.  Re-running is a no-op.
"""

import argparse
import os
import sys
from pathlib import Path

THIRD_PARTY = Path(os.environ.get("THIRD_PARTY", Path(__file__).resolve().parents[2]))

PATCHES = [
    dict(
        name="fragfm-reconstruct-remove-h",
        path=THIRD_PARTY / "FragFM" / "fragfm" / "utils" / "mol_ops.py",
        why=(
            "Upstream bug at HEAD (dcc6652): process.py calls "
            "reconstruct_to_rdmol(..., remove_h=...) but the function has no such "
            "argument, so fragfm's entire preprocessing path raises TypeError. "
            "remove_h=False callers match the result against AddHs() output by atom "
            "index, so that path must skip both RemoveHs and the SMILES round-trip "
            "in valid_mol_can_with_seg (which reorders atoms)."
        ),
        old="""def reconstruct_to_rdmol(h, e_index, e, is_relaxed=False, get_largest=True, fix=False):""",
        new="""def reconstruct_to_rdmol(
    h, e_index, e, is_relaxed=False, get_largest=True, fix=False, remove_h=True
):""",
    ),
    dict(
        name="fragfm-reconstruct-remove-h-body",
        path=THIRD_PARTY / "FragFM" / "fragfm" / "utils" / "mol_ops.py",
        why="Second half of the above: honour remove_h at the return site.",
        # `old` survives inside `new`, so anchor absence cannot detect this one.
        marker="if not remove_h:",
        old="""    if get_largest:
        # get largest connected component
        mol = valid_mol_can_with_seg(mol, largest_connected_comp=get_largest)
    else:
        mol = mol.GetMol()

    assert mol is not None
    mol = Chem.RemoveHs(mol)

    return mol""",
        new="""    if not remove_h:
        # Callers passing remove_h=False index into this molecule to build a
        # reordering map, so neither RemoveHs nor the SMILES round-trip inside
        # valid_mol_can_with_seg may run: both would change the atom order.
        mol = mol.GetMol()
        assert mol is not None
        Chem.SanitizeMol(mol, catchErrors=True)
        return mol

    if get_largest:
        # get largest connected component
        mol = valid_mol_can_with_seg(mol, largest_connected_comp=get_largest)
    else:
        mol = mol.GetMol()

    assert mol is not None
    mol = Chem.RemoveHs(mol)

    return mol""",
    ),
    dict(
        name="fragfm-spectrum-conditioning",
        path=THIRD_PARTY / "FragFM" / "fragfm" / "model" / "flow.py",
        why=(
            "Add an optional conditioning vector to the coarse GNN.  g_embd is "
            "built once from the fragment bag, the timestep and the latent, then "
            "carried through every layer, so a fourth slot reaches every node at "
            "every depth.  Off by default, so unconditional FragFM is unchanged."
        ),
        marker="use_spectrum_cond",
        old="""        # global feature embedder
        self.merge_embd_g = MLP(
            [cfg.embd_h_dim * 3, cfg.embd_h_dim, cfg.embd_h_dim],""",
        new="""        # global feature embedder
        self.merge_embd_g = MLP(
            [cfg.embd_h_dim * (4 if cfg.get("use_spectrum_cond", False) else 3),
             cfg.embd_h_dim, cfg.embd_h_dim],""",
    ),
    dict(
        name="fragfm-spectrum-conditioning-sig",
        path=THIRD_PARTY / "FragFM" / "fragfm" / "model" / "flow.py",
        why="Second part: accept the vector.",
        marker="cond=None,",
        old="""        timestep,
        frag_zs,
        coarse_h_valency=None,
    ):""",
        new="""        timestep,
        frag_zs,
        coarse_h_valency=None,
        cond=None,
    ):""",
    ),
    dict(
        name="fragfm-spectrum-conditioning-cat",
        path=THIRD_PARTY / "FragFM" / "fragfm" / "model" / "flow.py",
        why="Third part: concatenate it into the global feature.",
        marker="g_embd_parts",
        old="""        g_embd = torch.cat([frag_bag_embd_, timestep_embd, latent_z_embd], dim=1)""",
        new="""        g_embd_parts = [frag_bag_embd_, timestep_embd, latent_z_embd]
        if cond is not None:
            g_embd_parts.append(cond)
        g_embd = torch.cat(g_embd_parts, dim=1)""",
    ),
    dict(
        name="trainflow-cond-signature",
        path=THIRD_PARTY / "FragFM" / "exe" / "train_flow.py",
        why=(
            "Let process_single_epoch take a conditioning model.  We import and "
            "reuse this function rather than transcribe it: the corruption logic "
            "(antithetic time sampling, per-prior masking schedules, the "
            "fragment-bag mask) is subtle, and a transcription error would not "
            "raise, it would just train worse."
        ),
        marker="cond_model=None,",
        old="""    frag_occurance_source="train",
    optimizer=None,
):""",
        new="""    frag_occurance_source="train",
    optimizer=None,
    cond_model=None,
):""",
    ),
    dict(
        name="trainflow-cond-forward",
        path=THIRD_PARTY / "FragFM" / "exe" / "train_flow.py",
        why="Second part: encode the spectrum riding on the coarse graph and pass it in.",
        marker="cond_inputs(coarse_graph)",
        old="""        pred_h_embd, pred_e_logit, pred_z = coarse_gnn(
            ht_onehot,
            coarse_graph.full_e_index,
            et_onehot,
            zt,
            coarse_graph.batch,
            model_t,
            frag_zs,
            h_valency,
        )""",
        new="""        cond = None
        if cond_model is not None:
            from msfragfm.spectra_data import cond_inputs

            cond = cond_model(cond_inputs(coarse_graph))

        pred_h_embd, pred_e_logit, pred_z = coarse_gnn(
            ht_onehot,
            coarse_graph.full_e_index,
            et_onehot,
            zt,
            coarse_graph.batch,
            model_t,
            frag_zs,
            h_valency,
            cond=cond,
        )""",
    ),
    dict(
        name="trainflow-cond-clip",
        path=THIRD_PARTY / "FragFM" / "exe" / "train_flow.py",
        why="Third part: clip the conditioning model's gradients with the rest.",
        marker="clip_grad_norm_(cond_model.parameters()",
        old="""            torch.nn.utils.clip_grad_norm_(frag_embedder.parameters(), cfg.grad_clip)
            optimizer.step()""",
        new="""            torch.nn.utils.clip_grad_norm_(frag_embedder.parameters(), cfg.grad_clip)
            if cond_model is not None:
                torch.nn.utils.clip_grad_norm_(cond_model.parameters(), cfg.grad_clip)
            optimizer.step()""",
    ),
    dict(
        name="trainflow-cond-ema",
        path=THIRD_PARTY / "FragFM" / "exe" / "train_flow.py",
        why=(
            "EMA the conditioning model along with the rest.  FragFM's generator "
            "loads EMA weights (`frag_embedder_ema_*.pt`), so a spectrum encoder "
            "left out of the average would be mismatched with the flow it was "
            "trained beside."
        ),
        marker="ema_cond_model",
        old="""                update_ema(frag_embedder, ema_frag_embedder)
                update_ema(coarse_gnn, ema_coarse_gnn)""",
        new="""                update_ema(frag_embedder, ema_frag_embedder)
                update_ema(coarse_gnn, ema_coarse_gnn)
                if cond_model is not None:
                    update_ema(cond_model, ema_cond_model)""",
    ),
    dict(
        name="fragfm-generator-cond",
        path=THIRD_PARTY / "FragFM" / "fragfm" / "mol_generator.py",
        why=(
            "Pass a conditioning vector to the coarse GNN at generation time. "
            "Read off the sampler as an attribute rather than threaded through "
            "sample_molecule_graph_dynamic and _calc_euler_step, so the diff is "
            "one line and the unconditional path is untouched."
        ),
        marker="getattr(self, \"cond\", None)",
        old="""                cur_frag_zs,
                gen_h_valency,
            )""",
        new="""                cur_frag_zs,
                gen_h_valency,
                cond=getattr(self, "cond", None),
            )""",
    ),
    dict(
        name="fragfm-formula-support-mask",
        path=THIRD_PARTY / "FragFM" / "fragfm" / "mol_generator.py",
        why=(
            "Enforce the formula in the support.  frag_mask already gates which "
            "fragments a node may take, so intersecting it with a per-molecule "
            "admissibility mask makes compositionally impossible fragments "
            "unreachable rather than merely unlikely.  Read off the sampler as an "
            "attribute, so the unconditional path is untouched.  The any() guard "
            "matters: a node with every candidate masked would give all -inf "
            "logits and a NaN softmax, so such nodes keep the unconstrained mask."
        ),
        marker="frag_admissible",
        old="""            frag_mask = frag_mask[:, :-1].bool()  # exc. M""",
        new="""            frag_mask = frag_mask[:, :-1].bool()  # exc. M
            adm = getattr(self, "frag_admissible", None)
            if adm is not None:
                # [bs, n_all_frag] -> [bs, n_cur_frag] -> [n_node, n_cur_frag];
                # narrow to the bag first, or the intermediate is n_node x pool.
                node_adm = adm[:, cur_frag_idxs[:-1]][graph.batch]
                constrained = frag_mask & node_adm
                frag_mask = torch.where(
                    constrained.any(dim=1, keepdim=True), constrained, frag_mask
                )""",
    ),
    dict(
        name="fragfm-track-components",
        path=THIRD_PARTY / "FragFM" / "fragfm" / "mol_generator.py",
        why=(
            "Count components before truncation.  reconstruct_to_rdmol is called "
            "with get_largest=True, which keeps only the largest connected "
            "component, and the following `assert not '.' in smi` then passes -- "
            "so a disconnected assembly is recorded as a valid molecule rather "
            "than rejected.  Measured, 79% of the heavy-atom shortfall is mass "
            "discarded here.  Off unless the sampler asks for it."
        ),
        marker="track_components",
        old="""                m = reconstruct_to_rdmol(
                    h, e_index, e, is_relaxed=is_relaxed, get_largest=True
                )""",
        new="""                if getattr(self, "track_components", False):
                    try:
                        _raw = reconstruct_to_rdmol(
                            h, e_index, e, is_relaxed=is_relaxed, get_largest=False
                        )
                        self.component_counts.append(
                            Chem.MolToSmiles(_raw).count(".") + 1
                        )
                    except Exception:
                        self.component_counts.append(-1)
                m = reconstruct_to_rdmol(
                    h, e_index, e, is_relaxed=is_relaxed, get_largest=True
                )""",
    ),
    dict(
        name="fragfm-cache-frag-embeddings",
        path=THIRD_PARTY / "FragFM" / "fragfm" / "mol_generator.py",
        why=(
            "Embedding the pool costs 1m47s at every startup -- 83,194 fragments "
            "in chunks of 200, where building each chunk's graphs on CPU "
            "dominates the GPU work.  The embedder is frozen at generation, so "
            "the result depends only on its weights and the pool.  Keyed on the "
            "embedder's own tensors, so a different checkpoint cannot silently "
            "reuse another's embeddings."
        ),
        marker="_frag_z_cache",
        old="""        all_frag_idxs = torch.arange(self.n_all_frag)""",
        new="""        import hashlib
        import os as _os

        _key = hashlib.md5(repr(sorted(
            (k, tuple(v.shape), round(float(v.double().sum()), 6))
            for k, v in frag_embedder_sd.items()
        )).encode() + str(self.n_all_frag).encode()).hexdigest()[:16]
        self._frag_z_cache = _os.path.join(
            _os.path.dirname(str(self.cfg.frag_data_dirn)), f"_frag_z_{_key}.pt"
        )
        self._frag_z_cached = _os.path.exists(self._frag_z_cache)
        if self._frag_z_cached:
            _c = torch.load(self._frag_z_cache, map_location="cuda")
            self.all_frag_z, self.all_frag_junction_count = _c["z"], _c["jc"]
            print(f"Fragment embeddings from cache: {self._frag_z_cache}")

        all_frag_idxs = torch.arange(self.n_all_frag)""",
    ),
    dict(
        name="fragfm-cache-frag-embeddings-skip",
        path=THIRD_PARTY / "FragFM" / "fragfm" / "mol_generator.py",
        why="Second part: an empty range skips the loop on a cache hit.",
        marker="0 if self._frag_z_cached else self.n_all_frag",
        old="""            range(0, self.n_all_frag, 200),""",
        new="""            range(0, 0 if self._frag_z_cached else self.n_all_frag, 200),""",
    ),
    dict(
        name="fragfm-cache-frag-embeddings-write",
        path=THIRD_PARTY / "FragFM" / "fragfm" / "mol_generator.py",
        why="Third part: keep the cached tensors on a hit, write them on a miss.",
        marker="if not self._frag_z_cached:",
        old="""        self.all_frag_z = torch.cat(frag_z_list, dim=0).detach()
        self.all_frag_junction_count = torch.cat(
            frag_junction_count_list, dim=0
        ).detach()""",
        new="""        if not self._frag_z_cached:
            self.all_frag_z = torch.cat(frag_z_list, dim=0).detach()
            self.all_frag_junction_count = torch.cat(
                frag_junction_count_list, dim=0
            ).detach()
            torch.save({"z": self.all_frag_z,
                        "jc": self.all_frag_junction_count}, self._frag_z_cache)""",
    ),
    dict(
        name="fragfm-tree-assembly",
        path=THIRD_PARTY / "FragFM" / "fragfm" / "genererate_utils.py",
        why=(
            "Realise the coarse tree at atom level.  Blossom maximises matching "
            "weight over junction slots and is under no obligation to produce one "
            "bond per coarse edge, so 64.7% of assemblies came out disconnected "
            "and get_largest=True discarded the remainder (R21).  Overrides the "
            "matching result rather than replacing the call, so a failed "
            "assignment falls back to the released behaviour instead of "
            "producing something worse."
        ),
        marker="msfragfm.assemble",
        old="""        sel_recon_ae_to_pred_e_type = pred_ae_adj[
            d.ae_to_pred_index[0], d.ae_to_pred_index[1]
        ]""",
        new="""        sel_recon_ae_to_pred_e_type = pred_ae_adj[
            d.ae_to_pred_index[0], d.ae_to_pred_index[1]
        ]

        from msfragfm import assemble as _asm

        if _asm.ENABLED:
            _alt = _asm.assemble_tree(d)
            if _alt is not None:
                sel_recon_ae_to_pred_e_type = _alt.to(
                    sel_recon_ae_to_pred_e_type.dtype
                )""",
    ),
    dict(
        name="fragfm-constrained-final-step",
        path=THIRD_PARTY / "FragFM" / "fragfm" / "mol_generator.py",
        why=(
            "Three exact constraints at the last Euler step, in the only order "
            "that works.  Edges first: the spanning tree fixes each node's "
            "degree, and the degree is what makes the valency constraint "
            "expressible at all.  Then node types masked to fragments whose "
            "junction_count equals that degree, then the composition projection "
            "over what survives.  "
            "The valency constraint is the one that matters.  Measured, 85.6% of "
            "nodes get a fragment whose slot count matches their degree -- and a "
            "molecule needs every node to match, so 0.86^7 leaves 38% of "
            "molecules connected, which is exactly the 0.384 observed.  A 14% "
            "per-node error compounding over 7 nodes is a 62% molecule-level "
            "failure.  FragFM computes this mismatch as a feature "
            "(h_valency = h_degree - h_junction_count) and never constrains it."
        ),
        marker="enforce_valency",
        old="""        if is_last:
            gen_h_type = torch.argmax(pred_h1_prob, dim=1)
            glob_gen_h_type = cur_frag_idxs.to(device)[gen_h_type]
            gen_e_type = torch.argmax(pred_e1_prob, dim=1)
            gen_z = pred_z
            return glob_gen_h_type, gen_e_type, gen_z""",
        new="""        if is_last:
            # Edges first: the tree fixes each node's degree, and the degree is
            # what the valency constraint on node types is stated against.
            gen_e_type = torch.argmax(pred_e1_prob, dim=1)
            if getattr(self, "spanning_tree_decode", False):
                from msfragfm.spanning_tree import max_weight_spanning_tree

                bond_p = torch.softmax(pred_e_logit, dim=1)[:, 1]
                gen_e_type = max_weight_spanning_tree(
                    bond_p, full_e_index, batch
                ).long()

            node_logit = pred_h_logit[:, :n_cur_frag].clone()
            if getattr(self, "enforce_valency", False):
                deg = torch.zeros(node_logit.size(0), device=device)
                sel = full_e_index[:, gen_e_type == 1]
                ones = torch.ones(sel.size(1), device=device)
                deg.index_add_(0, sel[0], ones)
                deg.index_add_(0, sel[1], ones)
                cand_jc = self.all_frag_junction_count.to(device)[
                    cur_frag_idxs[:n_cur_frag]
                ].float()
                ok = cand_jc.unsqueeze(0) == deg.unsqueeze(1)
                # A node with no matching candidate keeps its unconstrained
                # options: all -inf would give a NaN softmax downstream.
                keep = ok.any(dim=1, keepdim=True)
                node_logit = node_logit.masked_fill(keep & ~ok, float("-inf"))

            gen_h_type = torch.argmax(node_logit, dim=1)
            targets = getattr(self, "composition_targets", None)
            if targets is not None:
                from msfragfm.composition import project_batch

                gen_h_type, frac = project_batch(
                    node_logit, batch, cur_frag_idxs[:n_cur_frag],
                    self.composition_counts, targets,
                )
                self.last_projection_rate = frac

            glob_gen_h_type = cur_frag_idxs.to(device)[gen_h_type]
            gen_z = pred_z
            return glob_gen_h_type, gen_e_type, gen_z""",
    ),
    dict(
        name="fragfm-cross-attention-init",
        path=THIRD_PARTY / "FragFM" / "fragfm" / "model" / "flow.py",
        why=(
            "Per-node cross-attention to the spectrum.  R24 measured fragment "
            "recall at 28.5% against 80%+ teacher-forced, and the conditioning "
            "arrives as a single pooled vector shared across all ~7 slots and "
            "distributed only by message passing.  Deciding which fragment a peak "
            "implies is per-node retrieval, which is what cross-attention is for "
            "and what Method A specifies.  One block before the backbone rather "
            "than one per layer: cheaper, and enough to test whether the "
            "granularity is the problem.  Gated on its own flag rather than "
            "use_spectrum_cond, which the existing checkpoint already has set "
            "-- sharing the flag would create these parameters when loading it "
            "and fail on missing keys, and it also keeps the two conditioning "
            "mechanisms independently ablatable."
        ),
        marker="spectrum_attn",
        old="""        # fragment bag embedder (optional)""",
        new="""        if cfg.get("use_cross_attention", False):
            self.spectrum_attn = torch.nn.MultiheadAttention(
                cfg.embd_h_dim, cfg.backbone_n_head, batch_first=True
            )
            self.spectrum_norm = torch.nn.LayerNorm(cfg.embd_h_dim)

        # fragment bag embedder (optional)""",
    ),
    dict(
        name="fragfm-cross-attention-forward",
        path=THIRD_PARTY / "FragFM" / "fragfm" / "model" / "flow.py",
        why="Second part: attend from node embeddings to the spectrum memory.",
        marker="cond_mem=None",
        old="""        coarse_h_valency=None,
        cond=None,
    ):""",
        new="""        coarse_h_valency=None,
        cond=None,
        cond_mem=None,
        cond_keep=None,
    ):""",
    ),
    dict(
        name="fragfm-cross-attention-apply",
        path=THIRD_PARTY / "FragFM" / "fragfm" / "model" / "flow.py",
        why=(
            "Third part: applied after the node features exist and before the "
            "backbone, so every message-passing layer sees spectrum-informed "
            "nodes.  Residual, so an untrained attention starts as a no-op."
        ),
        marker="self.spectrum_attn(",
        old="""        # make edge bidirectral
        e_embd = self.embd_coarse_e(e)""",
        new="""        if cond_mem is not None and hasattr(self, "spectrum_attn"):
            # One query per node against its own molecule's spectrum tokens.
            q = self.spectrum_norm(h_embd).unsqueeze(1)
            mem = cond_mem[batch]
            pad = ~cond_keep[batch] if cond_keep is not None else None
            att, _ = self.spectrum_attn(q, mem, mem, key_padding_mask=pad,
                                        need_weights=False)
            h_embd = h_embd + att.squeeze(1)

        # make edge bidirectral
        e_embd = self.embd_coarse_e(e)""",
    ),
    dict(
        name="trainflow-cond-memory",
        path=THIRD_PARTY / "FragFM" / "exe" / "train_flow.py",
        why=(
            "Pass the per-token spectrum memory as well as the pooled vector, so "
            "the cross-attention block has something to attend to."
        ),
        marker="cond_model.memory(",
        old="""            cond = cond_model(cond_inputs(coarse_graph))

        pred_h_embd, pred_e_logit, pred_z = coarse_gnn(
            ht_onehot,
            coarse_graph.full_e_index,
            et_onehot,
            zt,
            coarse_graph.batch,
            model_t,
            frag_zs,
            h_valency,
            cond=cond,
        )""",
        new="""            _ci = cond_inputs(coarse_graph)
            cond = cond_model(_ci)
            cond_mem, cond_keep = cond_model.memory(_ci)

        pred_h_embd, pred_e_logit, pred_z = coarse_gnn(
            ht_onehot,
            coarse_graph.full_e_index,
            et_onehot,
            zt,
            coarse_graph.batch,
            model_t,
            frag_zs,
            h_valency,
            cond=cond,
            cond_mem=cond_mem if cond_model is not None else None,
            cond_keep=cond_keep if cond_model is not None else None,
        )""",
    ),
    dict(
        name="trainflow-frag-mask-dropout",
        path=THIRD_PARTY / "FragFM" / "exe" / "train_flow.py",
        why=(
            "Training always guarantees the answer is available.  frag_mask is "
            "base_frags union the molecule's own fragments, so the true fragment "
            "is selectable at every step; at generation it is available 74% of the "
            "time (R16).  The model has never had to cope with its absence, which "
            "is exposure bias with a specific cause.  With probability "
            "cfg.frag_mask_dropout the guarantee is withheld per molecule, and the "
            "loss then skips nodes whose target became unselectable -- asking a "
            "model to predict what it cannot choose would give an infinite CE."
        ),
        marker="frag_mask_dropout",
        old="""        frag_mask = frag_mask.bool() | temp_h_in_batch  # [bs, n_cur_frag]""",
        new="""        if cfg.get("frag_mask_dropout", 0.0) > 0.0:
            drop = (torch.rand(bs, device=device) < cfg.frag_mask_dropout)
            temp_h_in_batch = temp_h_in_batch & ~drop[coarse_graph.batch].unsqueeze(1)
        frag_mask = frag_mask.bool() | temp_h_in_batch  # [bs, n_cur_frag]""",
    ),
    dict(
        name="trainflow-skip-unreachable-targets",
        path=THIRD_PARTY / "FragFM" / "exe" / "train_flow.py",
        why=(
            "Second part: a node whose target was masked out has -inf at the "
            "target and would contribute infinite cross-entropy, so it is dropped "
            "from the loss rather than allowed to destroy the gradient."
        ),
        marker="reachable = torch.isfinite",
        old="""        h_loss = F.cross_entropy(pred_h_logit, h_type, reduction="mean")""",
        new="""        reachable = torch.isfinite(
            pred_h_logit.gather(1, h_type.unsqueeze(1)).squeeze(1)
        )
        if reachable.any():
            h_loss = F.cross_entropy(
                pred_h_logit[reachable], h_type[reachable], reduction="mean"
            )
        else:
            h_loss = pred_h_logit.sum() * 0.0""",
    ),
    dict(
        name="fragfm-generator-cond-memory",
        path=THIRD_PARTY / "FragFM" / "fragfm" / "mol_generator.py",
        why=(
            "Same cross-attention memory at generation.  Read off the sampler as "
            "attributes, matching how the pooled vector is already passed."
        ),
        marker="cond_mem=getattr(self",
        old="""                cond=getattr(self, "cond", None),
            )""",
        new="""                cond=getattr(self, "cond", None),
                cond_mem=getattr(self, "cond_mem", None),
                cond_keep=getattr(self, "cond_keep", None),
            )""",
    ),
    dict(
        name="fragfm-single-lmdb-open",
        path=THIRD_PARTY / "FragFM" / "fragfm" / "mol_generator.py",
        why=(
            "FragFMGenerator opens the fragment LMDB, then hands the same path to "
            "FragFMDataset, which opens it again.  py-lmdb 2.x refuses a second "
            "open of one environment in a process; the lmdb==1.5.1 that FragFM "
            "pins allowed it.  Build the dataset first and share its handle."
        ),
        marker="Reuse the dataset's handle",
        old="""        # get fragment lmdb env
        self.frag_env = lmdb.open(
            cfg.frag_data_dirn,
            readonly=True,
            lock=False,
            readahead=True,
            meminit=False,
            map_size=100000000,
        )
        self.n_all_frag = int(self.frag_env.stat()["entries"])  # exc. mask

        # get test set and loader
        self.test_set = FragFMDataset(
            lmdb_fn=self.cfg.data_dirn,
            frag_lmdb_fn=self.cfg.frag_data_dirn,
            frag_smi_to_idx_fn=self.cfg.frag_smi_to_idx_fn,
            data_split=self.cfg.fragment_bag,
            debug=self.cfg.debug,
        )""",
        new="""        # get test set and loader
        self.test_set = FragFMDataset(
            lmdb_fn=self.cfg.data_dirn,
            frag_lmdb_fn=self.cfg.frag_data_dirn,
            frag_smi_to_idx_fn=self.cfg.frag_smi_to_idx_fn,
            data_split=self.cfg.fragment_bag,
            debug=self.cfg.debug,
        )
        # Reuse the dataset's handle rather than opening a second environment on
        # the same path, which py-lmdb 2.x rejects.
        self.frag_env = self.test_set.frag_env
        self.n_all_frag = int(self.frag_env.stat()["entries"])  # exc. mask""",
    ),
    dict(
        name="fragfm-drop-unused-draw-import",
        paths=[
            THIRD_PARTY / "FragFM" / "process" / "process_fragment_from_lmdb.py",
            THIRD_PARTY / "FragFM" / "process" / "process_to_lmdb.py",
            THIRD_PARTY / "FragFM" / "exe" / "eval_ae.py",
            THIRD_PARTY / "FragFM" / "exe" / "train_ae.py",
        ],
        why=(
            "These four import rdkit.Chem.Draw and never use it.  Draw needs "
            "libXrender.so.1, absent from the cluster image and un-installable "
            "without root, so an unused import blocks the entire preprocessing "
            "and autoencoder path."
        ),
        old="from rdkit.Chem import QED, Crippen, Descriptors, Draw",
        new="from rdkit.Chem import QED, Crippen, Descriptors",
    ),
    dict(
        name="mspred-lazy-plot-import",
        path=THIRD_PARTY / "ms-pred" / "src" / "ms_pred" / "common" / "__init__.py",
        why=(
            "ms_pred.common eagerly imports plot_utils, which imports "
            "rdkit.Chem.Draw, which needs libXrender.so.1.  The cluster image "
            "lacks it and we cannot apt-install without root.  Nothing on the "
            "training or RL path draws molecules, so import it on demand."
        ),
        old="""from .plot_utils import plot_mol_as_vector, plot_compare_ms, plot_ms""",
        new='''_PLOT_NAMES = ("plot_mol_as_vector", "plot_compare_ms", "plot_ms")


def __getattr__(name):
    # plot_utils -> rdkit.Chem.Draw -> libXrender.so.1, absent on headless
    # cluster images.  Import only if a caller actually plots.
    if name in _PLOT_NAMES:
        from . import plot_utils

        return getattr(plot_utils, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")''',
    ),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report only, do not write")
    args = ap.parse_args()

    n_applied = n_already = 0
    for p in PATCHES:
        for path in p.get("paths", [p.get("path")]):
            src = path.read_text(encoding="utf-8")
            label = f"{p['name']}:{path.name}" if "paths" in p else p["name"]
            # Deciding "already applied" is fiddly in both directions: `new` can
            # be a prefix of `old` (deleting a trailing import), and `old` can be
            # a substring of `new` (wrapping a block, which leaves the anchor in
            # place and invites re-application on every run).  A patch that wraps
            # its anchor must therefore give an explicit `marker` unique to the
            # patched state; the rest are decided by anchor absence.
            marker = p.get("marker")
            applied = marker in src if marker else (p["old"] not in src and p["new"] in src)
            if applied:
                print(f"  already  {label}")
                n_already += 1
                continue
            if p["old"] not in src:
                print(f"  STALE    {label}: anchor not found in {path}")
                sys.exit(2)
            if args.check:
                print(f"  pending  {label}")
                continue
            path.write_text(src.replace(p["old"], p["new"], 1), encoding="utf-8")
            print(f"  applied  {label}")
            n_applied += 1

    print(f"{n_applied} applied, {n_already} already present")


if __name__ == "__main__":
    main()
