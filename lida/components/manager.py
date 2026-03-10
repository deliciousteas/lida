# Visualization manager class that handles the visualization of the data with the following methods

# summarize data given a df
# generate goals given a summary
# generate generate visualization specifications given a summary and a goal
# execute the specification given some data

import os
from typing import List, Union
import logging

import pandas as pd
from llmx import llm, TextGenerator
from lida.datamodel import Goal, Summary, TextGenerationConfig, Persona
from lida.utils import read_dataframe
from ..components.summarizer import Summarizer
from ..components.goal import GoalExplorer
from ..components.persona import PersonaExplorer
from ..components.executor import ChartExecutor
from ..components.viz import VizGenerator, VizEditor, VizExplainer, VizEvaluator, VizRepairer, VizRecommender

import lida.web as lida


logger = logging.getLogger("lida")
"""Manager是控制LLM沟通的class，接受参数为`TextGenerator`类，该类来自于llmx库，用于初始化AI模型。
该类的每个方法都写在同目录下的components文件夹的不同模块，每个都用class来管理，这样降低耦合度，每个模块只要关心自己的业务功能就好。
由于该项目为23年，只考虑了gpt3.5系列模型，现今该模型已经deprecated，也没有开放兼容openai模型的配置。"""

class Manager(object):
    def __init__(self, text_gen: TextGenerator = None) -> None:
        """
        Initialize the Manager object.

        Args:
            text_gen (TextGenerator, optional): Text generator object. Defaults to None.
        """

        self.text_gen = text_gen or llm()

        self.summarizer = Summarizer()
        self.goal = GoalExplorer()
        self.vizgen = VizGenerator()
        self.vizeditor = VizEditor()
        self.executor = ChartExecutor()
        self.explainer = VizExplainer()
        self.evaluator = VizEvaluator()
        self.repairer = VizRepairer()
        self.recommender = VizRecommender()
        self.data = None
        self.infographer = None
        self.persona = PersonaExplorer()
    """
    Managger实例因为支持AI模型切换，需要检测AI配置是否相同，于是有一个check_textgen函数。
    本文没有保存每次对话的上下文记忆。
    """
    def check_textgen(self, config: TextGenerationConfig):
        """
        Check if self.text_gen is the same as the config passed in. If not, update self.text_gen.

        Args:
            config (TextGenerationConfig): Text generation configuration.
        """
        if config.provider is None:
            config.provider = self.text_gen.provider or "openai"
            logger.info("Provider is not set, using default provider - %s", config.provider)
            return

        if self.text_gen.provider != config.provider:

            logger.info(
                "Switching Text Generator Provider from %s to %s",
                self.text_gen.provider,
                config.provider)
            self.text_gen = llm(provider=config.provider)

    """
    该方法值得注意的是对数据做summary不仅仅是对表头、数据类型、统计分布的取值。
    在论文中，同样的数据input，具有summary环节能够显著降低代码编译错误率。
    semantic_type 和 summary的字段是发挥AI涌现能力的地方，基于summary+enrich_sementic字段补充，可以微调AI代码编译能力。
    参数列表：
        data参数可选，可以是一个str的路径名，也可以是dataframe格式，Union表示对类型的选择。
        n_samples参数指的是summary采样的数目，例如我取所有列的前三行做一个data summary。
        summary_method:为总结方法,默认生成数据的文件和统计信息，如果是llm则会有两阶段总结，第一阶段生成数据的文件和统计信息，第二阶段基于第一阶段的总结生成一个更丰富的总结，包含semantic_type和description字段。
        textgen_config:为生成summary的AI模型配置，例如temperature，top_p等。在openai api文档可查询。
    """
    def summarize(
        self,
        data: Union[pd.DataFrame, str],
        file_name="",
        n_samples: int = 3,
        summary_method: str = "default",
        textgen_config: TextGenerationConfig = TextGenerationConfig(n=1, temperature=0),
    ) -> Summary:
        """
        Summarize data given a DataFrame or file path.

        Args:
            data (Union[pd.DataFrame, str]): Input data, either a DataFrame or file path.
            file_name (str, optional): Name of the file if data is loaded from a file path. Defaults to "".
            n_samples (int, optional): Number of summary samples to generate. Defaults to 3.
            summary_method (str, optional): Summary method to use. Defaults to "default".
            textgen_config (TextGenerationConfig, optional): Text generation configuration. Defaults to TextGenerationConfig(n=1, temperature=0).

        Returns:
            Summary: Summary object containing the generated summary.

        Example of Summary:

            {'name': 'cars.csv',
            'file_name': 'cars.csv',
            'dataset_description': '',
            'fields': [{'column': 'Name',
            'properties': {'dtype': 'string',
                'samples': ['Nissan Altima S 4dr',
                'Mercury Marauder 4dr',
                'Toyota Prius 4dr (gas/electric)'],
                'num_unique_values': 385,
                'semantic_type': '',
                'description': ''}},
            {'column': 'Type',
            'properties': {'dtype': 'category',
                'samples': ['SUV', 'Minivan', 'Sports Car'],
                'num_unique_values': 5,
                'semantic_type': '',
                'description': ''}},
            {'column': 'AWD',
            'properties': {'dtype': 'number',
                'std': 0,
                'min': 0,
                'max': 1,
                'samples': [1, 0],
                'num_unique_values': 2,
                'semantic_type': '',
                'description': ''}},
            }

        """
        self.check_textgen(config=textgen_config)

        if isinstance(data, str):
            file_name = data.split("/")[-1]
            data = read_dataframe(data)

        self.data = data
        return self.summarizer.summarize(
            data=self.data, text_gen=self.text_gen, file_name=file_name, n_samples=n_samples,
            summary_method=summary_method, textgen_config=textgen_config)
    """
    goals 方法认可当前模型的图表生成能力，使用prompt engineering的方式引导模型生成符合数据分析师视角的、具有洞察力的图表生成目标。
    事实证明在2023年gpt-3系列对图表生成的能力已经有不错的能力，基于summary - goal - visualization的工作流，可以设计出很好的数据自动可视化系统。
    """
    def goals(
        self,
        summary: Summary,
        textgen_config: TextGenerationConfig = TextGenerationConfig(),
        n: int = 5,
        persona: Persona = None
    ) -> List[Goal]:
        """
        Generate goals based on a summary.

        Args:
            summary (Summary): Input summary.
            textgen_config (TextGenerationConfig, optional): Text generation configuration. Defaults to TextGenerationConfig().
            n (int, optional): Number of goals to generate. Defaults to 5.
            persona (Persona, str, dict, optional): Persona information. Defaults to None.

        Returns:
            List[Goal]: List of generated goals.

        Example of list of goals:

            Goal 0
            Question: What is the distribution of Retail_Price?

            Visualization: histogram of Retail_Price

            Rationale: This tells about the spread of prices of cars in the dataset.

            Goal 1
            Question: What is the distribution of Horsepower_HP_?

            Visualization: box plot of Horsepower_HP_

            Rationale: This tells about the distribution of horsepower of cars in the dataset.
        """
        self.check_textgen(config=textgen_config)

        if isinstance(persona, dict):
            persona = Persona(**persona)
        if isinstance(persona, str):
            persona = Persona(persona=persona, rationale="")

        return self.goal.generate(summary=summary, text_gen=self.text_gen,
                                  textgen_config=textgen_config, n=n, persona=persona)

    def personas(
            self, summary, textgen_config: TextGenerationConfig = TextGenerationConfig(),
            n=5):
        self.check_textgen(config=textgen_config)

        return self.persona.generate(summary=summary, text_gen=self.text_gen,
                                     textgen_config=textgen_config, n=n)
    """"
    visualizat 方法会使用summary 和 goal作为输入，然后将图表生成分为generate和execute2步骤。
    第一步：generate基于library参数生成符合相关代码；然后再编译执行。
    """
    def visualize(
        self,
        summary,
        goal,
        textgen_config: TextGenerationConfig = TextGenerationConfig(),
        library="seaborn",
        return_error: bool = False,
    ):
        if isinstance(goal, dict):
            goal = Goal(**goal)
        if isinstance(goal, str):
            goal = Goal(question=goal, visualization=goal, rationale="")

        self.check_textgen(config=textgen_config)
        code_specs = self.vizgen.generate(
            summary=summary, goal=goal, textgen_config=textgen_config, text_gen=self.text_gen,
            library=library)
        charts = self.execute(
            code_specs=code_specs,
            data=self.data,
            summary=summary,
            library=library,
            return_error=return_error,
        )
        return charts

    def execute(
        self,
        code_specs,
        data,
        summary: Summary,
        library: str = "seaborn",
        return_error: bool = False,
    ):

        if data is None:
            root_file_path = os.path.dirname(os.path.abspath(lida.__file__))
            print(root_file_path)
            data = read_dataframe(
                os.path.join(root_file_path, "files/data", summary.file_name)
            )

        # col_properties = summary.properties

        return self.executor.execute(
            code_specs=code_specs,
            data=data,
            summary=summary,
            library=library,
            return_error=return_error,
        )

    def edit(
        self,
        code,
        summary: Summary,
        instructions: List[str],
        textgen_config: TextGenerationConfig = TextGenerationConfig(),
        library: str = "seaborn",
        return_error: bool = False,
    ):
        """Edit a visualization code given a set of instructions

        Args:
            code (_type_): _description_
            instructions (List[Dict]): A list of instructions

        Returns:
            _type_: _description_
        """

        self.check_textgen(config=textgen_config)

        if isinstance(instructions, str):
            instructions = [instructions]

        code_specs = self.vizeditor.generate(
            code=code,
            summary=summary,
            instructions=instructions,
            textgen_config=textgen_config,
            text_gen=self.text_gen,
            library=library,
        )

        charts = self.execute(
            code_specs=code_specs,
            data=self.data,
            summary=summary,
            library=library,
            return_error=return_error,
        )
        return charts

    def repair(
        self,
        code,
        goal: Goal,
        summary: Summary,
        feedback,
        textgen_config: TextGenerationConfig = TextGenerationConfig(),
        library: str = "seaborn",
        return_error: bool = False,
    ):
        """ Repair a visulization given some feedback"""
        self.check_textgen(config=textgen_config)
        code_specs = self.repairer.generate(
            code=code,
            feedback=feedback,
            goal=goal,
            summary=summary,
            textgen_config=textgen_config,
            text_gen=self.text_gen,
            library=library,
        )
        charts = self.execute(
            code_specs=code_specs,
            data=self.data,
            summary=summary,
            library=library,
            return_error=return_error,
        )
        return charts
    """
    将可视化code输入给llm，基于prompt engineering 解释出来。
    """
    def explain(
        self,
        code,
        textgen_config: TextGenerationConfig = TextGenerationConfig(),
        library: str = "seaborn",
    ):
        """Explain a visualization code given a set of instructions

        Args:
            code (_type_): _description_
            instructions (List[Dict]): A list of instructions

        Returns:
            _type_: _description_
        """
        self.check_textgen(config=textgen_config)
        return self.explainer.generate(
            code=code,
            textgen_config=textgen_config,
            text_gen=self.text_gen,
            library=library,
        )

    def evaluate(
        self,
        code,
        goal: Goal,
        textgen_config: TextGenerationConfig = TextGenerationConfig(),
        library: str = "seaborn",
    ):
        """Evaluate a visualization code given a goal

        Args:
            code (_type_): _description_
            goal (Goal): A visualization goal

        Returns:
            _type_: _description_
        """

        self.check_textgen(config=textgen_config)

        return self.evaluator.generate(
            code=code,
            goal=goal,
            textgen_config=textgen_config,
            text_gen=self.text_gen,
            library=library,
        )

    def recommend(
        self,
        code,
        summary: Summary,
        n=4,
        textgen_config: TextGenerationConfig = TextGenerationConfig(),
        library: str = "seaborn",
        return_error: bool = False,
    ):
        """Edit a visualization code given a set of instructions

        Args:
            code (_type_): _description_
            instructions (List[Dict]): A list of instructions

        Returns:
            _type_: _description_
        """

        self.check_textgen(config=textgen_config)

        code_specs = self.recommender.generate(
            code=code,
            summary=summary,
            n=n,
            textgen_config=textgen_config,
            text_gen=self.text_gen,
            library=library,
        )
        charts = self.execute(
            code_specs=code_specs,
            data=self.data,
            summary=summary,
            library=library,
            return_error=return_error,
        )
        return charts

    def infographics(self, visualization: str, n: int = 1,
                     style_prompt: Union[str, List[str]] = "",
                     return_pil: bool = False
                     ):
        """
        Generate infographics using the peacasso package.

        Args:
            visualization (str): A visualization code
            n (int, optional): The number of infographics to generate. Defaults to 1.
            style_prompt (Union[str, List[str]], optional): A style prompt or list of style prompts. Defaults to "".

        Raises:
            ImportError: If the peacasso package is not installed.
        """

        try:
            import peacasso

        except ImportError as exc:
            raise ImportError(
                'Please install lida with infographics support. pip install lida[infographics]. You will also need a GPU runtime.'
            ) from exc

        from ..components.infographer import Infographer

        if self.infographer is None:
            logger.info("Initializing Infographer")
            self.infographer = Infographer()
        return self.infographer.generate(
            visualization=visualization, n=n, style_prompt=style_prompt, return_pil=return_pil)
