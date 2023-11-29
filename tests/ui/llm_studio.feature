Feature: LLMStudio

    Scenario: View datasets
        Given LLM Studio home page is opened
        When I login to LLM Studio
        Then I upload dataset using filesystem
        Then I should see the datasets

