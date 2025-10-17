# TextEncodersForSELD

<div align=center>
   <h2>
   Leveraging Pre-Trained Text Encoder for Sound Event Localization and Detection
   </h2>
</div>

## Abstract:
> Pre-trained language and multimodal models have emerged as powerful universal encoders, offering transferable embeddings that generalize effectively across diverse tasks. Although prior work has explored these models for spatial sound analysis, particularly in audio-visual contexts, their adaptation to sound event localization and detection (SELD) remains limited. In this paper, we leverage the representational power of large-scale pre-trained text encoders for SELD tasks. First, we transform spectro-spatial audio features from multichannel input into language-compatible token sequences. These comprise an audio token that captures event content, a direction-of-arrival (DOA) token representing directional cues, and a distance token capturing the source range. Subsequently, these tokens are processed by the text encoder to produce compact audio-driven embeddings that jointly capture event identity and spatial attributes. We evaluated our framework on the DCASE stereo SELD dataset, demonstrating consistent improvements over SELD baselines. In addition to improving SELD performance, the proposed framework produces embeddings that serve as structured representations for future multimodal extensions.

<div align=center>
   <img src="./images/seld.png" style="width:80%; height:auto;">
   	<figcaption>
	  	<br>Proposed audio-only SELD framework. Stereo audio features are mapped to three language compatible tokens (audio, DOA and distance) - which are processed by the frozen text encoder to generate an audio-driven embedding (\textbf{A}) which serves as a compact representation of both the event identity and its spatial attributes.
    	</figcaption>
</div>
