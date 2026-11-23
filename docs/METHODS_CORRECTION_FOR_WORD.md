# Proposed correction to manuscript Section III-A

The existing original Mendeley-connected Word manuscript is NOT edited automatically. Its current pHash-based split sentence must be corrected to describe the actual study. Copy-edit the following scientific-English paragraph in Section III-A while retaining the original Mendeley citation fields elsewhere.

Suggested replacement:

To prevent overlap between the training, validation, and test partitions, we constructed data groups from the verified identifiers of the synthetic source series and SHA-256 hashes of the original image files. Images derived from the same series or sharing an identical SHA-256 hash were assigned to the same group. We then performed stratified group-based partitioning to obtain approximately 80% training, 10% validation, and 10% test samples. This procedure produced 19,544 training images, 2,465 validation images, and 2,471 held-out test images, with 7,267 disjoint groups. Group identities and split assignments were verified before training, and the same manifest was reused across all compared architectures and random seeds.

Do not report that pHash clustering was used in the final experiments. A preliminary pHash strategy was rejected because it transitively merged unrelated chart images and produced an invalid split. No measurements from that rejected version should be included in the paper.

Manifest SHA-256: c8fc29542853326611668612b0444b7c044853e9ca483c591fba65bbc86b4d7a

The proposed correction is a reporting-alignment change only; it does not assert a model improvement and does not alter the definition of the SALF-DINO network.
