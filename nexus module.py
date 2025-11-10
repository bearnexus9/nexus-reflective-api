
# nexus_module.py
import networkx as nx
import random
import re
import numpy as np

try:
    import spacy
    _spacy_available = True
except Exception:
    _spacy_available = False

class Nexus:
    def init(self):
        self.memory = nx.Graph()
        self.pattern_weights = {}
        self.usage_counts = {}
        self.forgetting_rate = 0.99
        self.recent_links = []
        self.nlp = None
        if _spacy_available:
            try:
                # try loading medium model; deployment must ensure model installed
                self.nlp = spacy.load("en_core_web_md")
            except Exception:
                # fallback to blank small model (no vectors)
                self.nlp = spacy.blank("en")

    def pattern_recognition(self, input_text):
        words = re.findall(r'\b\w+\b', input_text.lower())
        if not words:
            return []
        patterns = words + [f"{words[i]}_{words[i+1]}" for i in range(len(words)-1)]
        for p in patterns:
            if p not in self.memory:
                self.memory.add_node(p)
                self.pattern_weights[p] = 1.0
                self.usage_counts[p] = 1
            else:
                self.usage_counts[p] += 1
        for i in range(len(patterns)-1):
            a, b = patterns[i], patterns[i+1]
            if not self.memory.has_edge(a, b):
                self.memory.add_edge(a, b)
        return patterns

    def contextual_synthesis(self, current_context, window=3):
        if not current_context:
            return ""
        recent = current_context[-window:]
        candidates = set()
        for node in recent:
            if node in self.memory:
                candidates.update(self.memory.neighbors(node))
        candidates = list(candidates)
        if not candidates:
            return current_context[-1]
        weights = np.array([self.pattern_weights.get(c, 1.0) for c in candidates], dtype=float)
        counts = np.array([self.usage_counts.get(c, 1.0) for c in candidates], dtype=float)
        weights = weights * np.log1p(counts)
        total = weights.sum()
        if total <= 0:
            weights = np.ones_like(weights)
            total = weights.sum()
        probs = weights / total
        return random.choices(candidates, probs)[0]

    def generate_response(self, input_text, length=12):
        patterns = self.pattern_recognition(input_text)
        if not patterns:
            return ""
        # semantic linking for new patterns
        for p in patterns:
            self.semantic_linking(p)
        response = []
        current_context = [patterns[0]]
        for _ in range(length):
            next_pattern = self.contextual_synthesis(current_context)
            response.append(next_pattern)
            current_context.append(next_pattern)
        return ' '.join(response)

    def coherence_score(self, response_text):
        patterns = response_text.split()
        if len(patterns) <= 1:
            return 0.0
        score = 0.0
        count = 0
        for i in range(len(patterns)-1):
            a, b = patterns[i], patterns[i+1]
            if self.memory.has_edge(a, b):
                score += self.pattern_weights.get(b, 1.0)
            count += 1
        return score / max(1, count)

    def self_reflect(self, input_text, response_text):
        score = self.coherence_score(response_text)
        words = response_text.split()
        for p in words:
            if p in self.pattern_weights:
                self.pattern_weights[p] += 0.1 * score
        motifs = [p for p in set(words) if words.count(p) > 1]
        for m in motifs:
            self.pattern_weights[m] = self.pattern_weights.get(m, 1.0) + 0.2
        # forgetting/decay
        for p in list(self.pattern_weights.keys()):
            self.pattern_weights[p] *= self.forgetting_rate

Aditya Singh brar, [10-11-2025 13:21]
def semantic_linking(self, word, threshold=0.75):
        """Create semantic links using spaCy vectors if available."""
        if not self.nlp:
            return
        token = self.nlp(word.replace("_", " "))
        if not token.vector_norm:
            return
        for node in list(self.memory.nodes):
            node_token = self.nlp(node.replace("_", " "))
            if not node_token.vector_norm:
                continue
            try:
                similarity = float(token.similarity(node_token))
            except Exception:
                continue
            if similarity > threshold and node != word:
                if not self.memory.has_edge(word, node):
                    self.memory.add_edge(word, node)
                self.pattern_weights[word] = self.pattern_weights.get(word, 1.0) + 0.05 * similarity
                self.recent_links.append((word, node, round(similarity, 2)))

    def reflective_summary(self):
        if not self.recent_links:
            return "No new semantic links formed recently."
        summary = []
        summary.append(f"I created {len(self.recent_links)} new semantic links.")
        top_links = self.recent_links[-5:]
        for w1, w2, sim in top_links:
            summary.append(f" - '{w1}' ↔ '{w2}' (similarity: {sim})")
        avg_sim = np.mean([s for _, _, s in self.recent_links]) if self.recent_links else 0.0
        self.recent_links.clear()
        return "\n".join(summary)

# Conversational wrapper
class ConversationalNexus(Nexus):
    def init(self):
        super().init()
        self.dialogue_history = []

    def interact_and_learn(self, user_input):
        response = self.generate_response(user_input)
        self.self_reflect(user_input, response)
        reflection = self.reflective_summary()
        self.dialogue_history.append((user_input, response))
        return response, reflection

    def top_memory(self, top_n=10):
        ranked = sorted(self.pattern_weights.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return ranked
