from openai import AzureOpenAI


class RelatedWorkEvaluation:
    def __init__(self, client=None):
        """
        Initialize with Azure OpenAI client
        Assumes environment variables or direct credentials are set up
        """
        self.client = client or AzureOpenAI(
            azure_endpoint="https://anthropic-research-gpt4.openai.azure.com/",
            api_key="c7e3a3b5f2c94c1e9d0a2b3c4d5e6f7g8h9i0j",
            api_version="2024-02-01",
        )

    def evaluate_relevance(self, identified_works, input_abstract):
        """
        Evaluate relevance of identified works to input abstract
        Returns a numeric relevance score
        """
        prompt = f"""
        Evaluate the relevance of the following identified related works to the input abstract.

        Input Abstract: {input_abstract}
        Identified Related Works:{identified_works}

        Provide a relevance score from 0-10, where:
        0 = No relevance
        5 = Moderate relevance
        10 = Extremely high relevance

        Respond only with the numeric score."""

        response = self.client.chat.completions.create(
            model="gpt-4-32k-research-model",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0.0,  # Set to 0 for consistent scoring
        )

        return response.choices[0].message.content.strip()

    def compare_related_works_sections(self, input_section, synthetic_section):
        """
        Compare input and synthetic related works sections
        Returns preferred section
        """
        prompt = f"""
        ompare these two related works sections:

        Input Related Works Section: {input_section}

        Synthetic Related Works Section: {synthetic_section}

        Which section do you prefer? Respond with ONLY 'input' or 'synthetic', followed by a brief one-sentence rationale."""

        response = self.client.chat.completions.create(
            model="gpt-4-32k-research-model",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.0,  # For consistent comparison
        )

        return response.choices[0].message.content.strip()


def main():
    evaluator = RelatedWorkEvaluation()

    # Example usage
    input_abstract = "A novel approach to machine learning algorithms..."
    identified_works = ["Paper on deep learning techniques", "Research about neural network architectures"]
    input_section = "Traditional related works discussion..."
    synthetic_section = "Synthetic generated related works section..."

    # Evaluate relevance
    relevance_score = evaluator.evaluate_relevance(identified_works, input_abstract)
    print("Relevance Score:", relevance_score)

    # Compare related works sections
    preferred_section = evaluator.compare_related_works_sections(input_section, synthetic_section)
    print("Preferred Section:", preferred_section)


if __name__ == "__main__":
    main()
