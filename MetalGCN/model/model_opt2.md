太好了，你的補充說明很完整。我會針對你目前模型的整體設計進行診斷，確認是否能準確模擬單一金屬與 1\~3 個相同配體組合的結合常數預測流程，並分析以下幾個面向：

* 預測綁定位點 pred\_pos 的可靠性與 validate\_binding\_sites 的作用
* 多配體結合位點與骨架特徵提取方式是否合理
* 對 true\_pka 序列與 output sequence 的對齊是否正確
* loss 與 RMSE 計算邏輯是否會產生錯位或不穩定
* sample 階段是否能輸出變長的結合常數序列

我會彙整這些分析並提供具體的修改建議與程式碼示範。


# Multi-Output Binding Prediction: Code Improvements and Loss Considerations

## Validation of Predicted Binding Sites (Code Review)

The provided `validate_binding_sites` function is designed to **improve predicted binding site positions** by enforcing chemical plausibility. It checks each predicted atom index and adjusts it if necessary:

* **Common Donor Atoms:** If a predicted atom is not one of the typical donor atoms for metal coordination (the code uses `{O, N, S, P, F, Cl, Br, I}`), it attempts to replace it with a neighboring atom that is a common donor (preferring O, N, S, or P). This makes sense because **metals usually bind to heteroatoms** like oxygen, nitrogen, sulfur, phosphorus, or halide ions rather than carbon or other atoms. For example, if the model predicted a carbon atom as a binding site, the function will look at that carbon's neighbors and pick the first neighbor that is O/N/S/P (if any) as a more likely binding site.

* **Ensuring Sufficient Neighboring Bonds:** If the predicted atom *is* a common donor atom, the function then checks its number of neighbors. The idea here is to avoid choosing an atom that is too “isolated.” For instance, a hydroxyl oxygen with only one bond might be less likely to coordinate a metal strongly compared to an oxygen that is part of a larger functional group. The parameter `min_neighbors` (default 1) sets the threshold. If the donor atom has fewer neighbors than `min_neighbors` (e.g., 0 in case of a completely isolated atom), the code searches the entire molecule for an alternative **well-connected donor** (specifically any oxygen or nitrogen with at least `min_neighbors` neighbors) and uses that instead. If no alternative is found, it keeps the original prediction.

Overall, this function acts as a post-processing step to **validate or correct binding site predictions**. It helps ensure that predicted binding sites correspond to chemically reasonable atoms (for example, N or O instead of a carbon) and are not isolated in a way that would make coordination chemistry unlikely. This kind of heuristic is practical in the absence of a more detailed chemical rule system. (One possible enhancement could be considering all neighbor candidates rather than just the first, or using chemical knowledge to pick the most likely coordinating atom among the neighbors, but the current approach is a good starting point.)

## Sorting of True pKa Values (Ascending Order)

Yes, sorting `batch.true_pka` from smallest to largest is a sensible approach given the context. If these values represent **binding constants or pK\_a values**, it is common to order them for consistency. In fact, for polyprotic acids (which have multiple pK\_a values), the convention is to list the pK\_a’s in increasing order (pK\_a1 is the first dissociation with the lowest pK\_a, pK\_a2 is higher, and so on, since each successive proton is harder to remove). By sorting the true values, you ensure a fixed order (for example, strongest binding or lowest pK\_a first) for both the targets and the model outputs.

From a modeling perspective, **sorting the targets solves the permutation problem**: the model doesn’t have to worry about which output corresponds to which constant, because both the prediction and ground truth can be compared in sorted order. This is especially important if the binding constants are inherently unlabeled. For instance, if a molecule has two binding sites with pK\_a of 6.5 and 8.0, sorting means we always treat 6.5 as "first" and 8.0 as "second". The model can then be trained to predict `[6.5, 8.0]` (in that order) for that molecule.

🔹 **Note:** Sorting is a straightforward heuristic to impose an order. An alternative (more advanced) approach in research is to use a *permutation-invariant loss function*, which finds the best match between predicted sets and true sets without requiring sorting. Such techniques (often used in set prediction or speech separation tasks) ensure that the loss is minimized over all permutations of the outputs. However, implementing a permutation-invariant loss is more complex. In our case, since you are already sorting the true values, you should also sort the model’s predicted values before computing the loss, so that each predicted pK\_a is aligned with the corresponding true pK\_a. This way, the loss calculation is consistent.

In summary, **yes, keep the true binding constants sorted**. It simplifies the learning problem by giving a consistent target order for the model. Just remember to apply the same ordering to predictions when evaluating. This choice will not affect the model’s ability to predict multiple values; it only affects how we measure error against the true values.

## Including All Outputs in the Loss Calculation

You are correct to include *all* predicted binding constants in the loss function. If the model outputs a set of values (e.g., multiple pK\_a’s or binding affinities for multiple sites/ligands), every one of those outputs should contribute to the training loss. By doing so, the model is encouraged to learn each value. **Omitting one or more outputs from the loss would effectively tell the model to ignore those outputs**, which is not what we want.

In practice, if you're using a framework like PyTorch or TensorFlow, this can be handled in a couple of ways:

* **Single Combined Loss:** If you use a single multi-output regression loss (e.g. Mean Squared Error that internally sums/averages over all output dimensions), it will naturally include all outputs. For example, in PyTorch, if `pred` and `true` are tensors of shape `[batch_size, num_outputs]`, `loss = mse_loss(pred, true)` will compute the average squared error across all outputs by default.

* **Multiple Losses Summed:** Alternatively, you can compute separate losses for each output and then sum them. This is mathematically equivalent to the above (possibly up to a normalization factor if you take a mean). For example, if your model returns `output1, output2, ...`, you could do something like:

  ```python
  loss1 = criterion(output1, target1)
  loss2 = criterion(output2, target2)
  total_loss = loss1 + loss2
  total_loss.backward()
  ```

  This approach was illustrated in the PyTorch forum – you simply add the losses from each output before backpropagation. The key is that **every output’s error is accounted for** in `total_loss`.

Including all outputs in the loss is important for the model’s performance. As you noted, this will also affect the model’s ability to **sample multiple binding constants during generation**. If, for instance, only the lowest pK\_a were used in the loss, the model might learn to only predict that one accurately and neglect the others. By training with all outputs, you ensure the model learns to predict the entire set of binding constants. This is crucial for when you want to sample/generate results: you want the model to output a full set of constants (e.g., a pair of values if there are two binding sites) rather than just one. In summary, **use all available data to train** – if a sample has N true values, make sure the error from all N predictions is counted.

## Implementing the Prediction Output for Multiple Binding Constants

Since you mentioned the prediction results are not yet written out, it’s important to implement a way to **extract and use the model’s predictions** after (or during) training:

* **During Training (for monitoring):** It can be helpful to print or log the model’s predictions for a batch of data periodically. This can be as simple as:

  ```python
  model.eval()
  with torch.no_grad():
      pred_values = model(batch_inputs)
      # pred_values might be a tensor of shape [batch_size, num_outputs]
      print(pred_values[0])  # print the predictions for the first sample in the batch
  model.train()
  ```

  This would show the predicted binding constants (and/or binding site indices, if your model has multiple outputs) for inspection. You might do this every few epochs to see if the predictions are moving in the right direction.

* **After Training (evaluation/inference):** To get final predictions, put the model in evaluation mode and loop through your test or validation dataset. Collect the predicted values for each sample. If the true values were sorted, remember to **sort the predicted values as well** (unless your model inherently outputs them in sorted order). Then you can compare each set of predicted constants to the true set (compute error metrics like MAE or RMSE for each output).

* **Formatting the Output:** Depending on your needs, you might want to save the predictions. For example, you can write the results to a CSV file or a JSON: for each molecule (or complex), list the predicted binding constants. Make sure to also output the corresponding true values for clarity. This makes it easier to analyze the model’s performance on each instance.

* **Using `validate_binding_sites` (if applicable):** If your model also predicts binding site positions (atom indices) along with the constants, you should run those predicted positions through the `validate_binding_sites` function before final output. This will correct any implausible site predictions. Then you can output the validated site indices together with the predicted constants. For instance, you might output something like: *“Molecule X: Predicted binding atoms = \[12, 45], Predicted pK\_a = \[6.5, 8.0], True pK\_a = \[6.3, 8.1]”*. This kind of report would let you see both what atoms were predicted as binding sites and what the binding constants were.

Implementing the prediction output is important for debugging and demonstration. Without actually examining the predictions, it’s hard to know how your model is doing beyond just a loss value. Once you write this part, you’ll be able to answer questions like: *Did the model predict both binding constants correctly for this molecule? Is it sometimes missing one of them? Is it predicting them in the wrong order?* Writing out the results will give insight into such issues.

## Handling Variable Number of Binding Outputs

Your dataset can have varying numbers of binding constants per sample (e.g., some molecules have 1, others have 2, etc., corresponding to the number of ligands or binding sites). You stated that in the data loading stage, this is handled such that if there are two ligands, there are two binding constants (and presumably they are paired correctly). The goal is: **if a sample has N binding sites (or ligands), the model should predict N binding constants.** This is a variable-output scenario, which can be tricky but is manageable with the right approach.

&#x20;*An illustration of a protein (Caspase-3) with two distinct binding sites (Site 1 and Site 2, shown in yellow and cyan surfaces) each capable of binding different ligands. Each site is bound to an example ligand (see the small molecules in the inset boxes). In such cases, each site–ligand pair has its own binding constant. A predictive model must therefore output multiple values (e.g., two binding affinity values) for a single target. Ensuring the model can handle a variable number of outputs is crucial for accurately modeling these multi-site systems.*

To ensure the model predicts as many values as needed for each sample, consider the following strategies:

* **Design the Model with a Fixed Maximum and Masking:** If there is an upper bound on how many binding constants any sample can have (say at most 3, for example), you can design the model to always output a fixed-size vector (of length 3 in this example). During training, you would supply a special mask or indicator for outputs that are not used. For instance, if a particular molecule has only 2 constants, you could pad the true output with a placeholder (or ignore the third output by not including it in the loss). In PyTorch, this could be done by multiplying the loss for that output by 0 or using an ignore index for classification, etc. However, since you mention no missing values in the dataloader, it sounds like you may be **grouping or batching data by the same number of outputs** or otherwise ensuring each sample exactly fits the model’s output size. If every sample in a batch has the same number of outputs, you effectively treat that as a fixed-size output for that batch.

* **Dynamic Model Output (Sequence modeling):** A more flexible approach is to allow the model to output a **variable-length sequence** of binding constants. This could be done with sequence models (like an RNN, transformer, or iterative prediction approach). For example, you could have the model first predict how many outputs (N) it should produce or take N as an input, and then generate N values one by one. If you supply the number of binding sites/ligands as an input feature to the model, the model can use that to know how many predictions to make. Concretely, you might encode the count N and feed it along with other features (some architectures allow conditioning on such auxiliary inputs). Then the model’s decoder could be run N times to output N values. In practice, this is more complex to implement and train — you have to ensure the model doesn’t generate more or fewer outputs than desired. But it aligns with the idea of *“if given N, predict N outputs.”*

* **One Output per Ligand (alternative data structuring):** Another conceptually simple approach is to restructure the problem so that each (ligand, binding constant) pair is treated as a separate training instance. In other words, rather than one sample with two outputs, you have two samples each with one output. However, this loses the notion that those two bindings happened to the same molecule or protein, and it doesn’t make the model explicitly learn a relationship between multiple outputs for one entity. Since your aim is clearly to have the model predict multiple outputs for a single entity, this restructuring is likely not what you want. I mention it just for completeness, as sometimes multi-output problems can be flattened into single-output ones at the cost of losing context.

Given your requirements, the first strategy (fixed maximum with masking) or a slight variant of it is usually easiest if the maximum number of outputs is small and known. For example, if at most 2 binding constants occur for any molecule in your dataset, you could have the model output `[value1, value2]` always. During training, for a molecule that truly has only one binding constant, you could let `value2` be a dummy (and not contribute to loss, or set its true value equal to the first so the model doesn’t get confused). But since you indicated the dataloader ensures matching numbers without missing values, you might already be handling this by not mixing different-length outputs in the same batch.

**Key point:** To achieve *“if given N, predict N”*, the model needs to be informed about N (explicitly or implicitly) **and** the training process should reinforce predicting exactly N values. If you feed a batch of all 2-output cases to a model with a 2-output head, it will naturally predict 2 values for each. If you then feed a batch of 1-output cases to a 1-output head model (or the same model configured somehow differently), it predicts 1 each. But a single model that can do both might require either dynamic architecture or padding.

Since you mention that at the data loading stage you don’t expect missing values, one practical approach could be: **train separate models for each case** (one model for 1-output, another for 2-outputs) or use a single model that has the maximum number of outputs and train it with appropriate masking. Most people try to avoid multiple separate models if the task is fundamentally the same, so masking is popular.

In summary, **make sure the model’s architecture and training procedure align with the variable-length nature of the output**:

* If you use a fixed-length output layer, handle the variable case by masking or only comparing the first N outputs to N targets.
* If you want a single model to truly vary its output length, consider sequence modeling techniques and give the model the information needed (like the count or an end-of-sequence token in other contexts).

By addressing this, when you provide an input that requires (say) two binding constants, the model will indeed output two values, and when an input only requires one, it will output just one. The fact that you’ve sorted the true values means the model doesn’t have to worry about which value is first or second, only to output the correct set. Ensuring the correct count is the final piece of the puzzle. With these considerations, you’ll be able to train a model that can handle multiple binding sites/ligands and predict a corresponding binding constant for each. Good luck with the implementation!

**Sources:**

1. Common donor atoms for metal coordination include N, O, S, P, and halogens (F, Cl, Br, I).
2. In polyprotic acids, pK\_a values are conventionally listed in ascending order (each subsequent pK\_a is higher).
3. Permutation-invariant loss functions can handle unordered outputs by not assuming any particular matching between output index and target index.
4. Example of handling multiple outputs in a model: computing separate losses for each and summing them for backpropagation.
