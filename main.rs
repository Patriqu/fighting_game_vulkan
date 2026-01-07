use cgmath::{Deg, Matrix4, Point3, Rad, Vector3, perspective};
use std::sync::Arc;
use vulkano::buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, RenderPassBeginInfo};
use vulkano::device::{Device, DeviceCreateInfo, QueueCreateInfo, QueueFlags};
use vulkano::image::view::ImageView;
use vulkano::image::{Image, ImageUsage};
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::graphics::color_blend::{ColorBlendAttachmentState, ColorBlendState};
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::RasterizationState;
use vulkano::pipeline::graphics::vertex_input::{Vertex, VertexDefinition};
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{GraphicsPipeline, PipelineLayout, PipelineShaderStageCreateInfo};
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass};
use vulkano::swapchain::{Surface, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo};
use vulkano::sync::{self, GpuFuture};
use vulkano::{Validated, VulkanError, VulkanLibrary};
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop, EventLoopWindowTarget};
use winit::window::WindowBuilder;

// Vertex structure for fighter characters
#[derive(BufferContents, Vertex, Clone, Copy)]
#[repr(C)]
struct FighterVertex {
    #[format(R32G32B32_SFLOAT)]
    position: [f32; 3],
    #[format(R32G32B32_SFLOAT)]
    normal: [f32; 3],
    #[format(R32G32_SFLOAT)]
    tex_coords: [f32; 2],
}

// Push constants for transformation matrices
#[derive(BufferContents, Clone, Copy)]
#[repr(C)]
struct PushConstants {
    model: [[f32; 4]; 4],
    view: [[f32; 4]; 4],
    projection: [[f32; 4]; 4],
}

// Fighter character state
struct Fighter {
    position: Vector3<f32>,
    rotation: f32,
    health: f32,
    is_attacking: bool,
    vertex_buffer: Arc<Subbuffer>,
    index_buffer: Arc<Subbuffer>,
}

impl Fighter {
    fn new(memory_allocator: Arc<StandardMemoryAllocator>, position: Vector3<f32>) -> Self {
        // Simple cube mesh for fighter (replace with actual character model)
        let vertices = vec![
            // Front face
            FighterVertex {
                position: [-0.5, -1.0, 0.5],
                normal: [0.0, 0.0, 1.0],
                tex_coords: [0.0, 0.0],
            },
            FighterVertex {
                position: [0.5, -1.0, 0.5],
                normal: [0.0, 0.0, 1.0],
                tex_coords: [1.0, 0.0],
            },
            FighterVertex {
                position: [0.5, 1.0, 0.5],
                normal: [0.0, 0.0, 1.0],
                tex_coords: [1.0, 1.0],
            },
            FighterVertex {
                position: [-0.5, 1.0, 0.5],
                normal: [0.0, 0.0, 1.0],
                tex_coords: [0.0, 1.0],
            },
            // Back face
            FighterVertex {
                position: [-0.5, -1.0, -0.5],
                normal: [0.0, 0.0, -1.0],
                tex_coords: [1.0, 0.0],
            },
            FighterVertex {
                position: [0.5, -1.0, -0.5],
                normal: [0.0, 0.0, -1.0],
                tex_coords: [0.0, 0.0],
            },
            FighterVertex {
                position: [0.5, 1.0, -0.5],
                normal: [0.0, 0.0, -1.0],
                tex_coords: [0.0, 1.0],
            },
            FighterVertex {
                position: [-0.5, 1.0, -0.5],
                normal: [0.0, 0.0, -1.0],
                tex_coords: [1.0, 1.0],
            },
        ];

        let indices: Vec<u16> = vec![
            0, 1, 2, 2, 3, 0, // Front
            5, 4, 7, 7, 6, 5, // Back
            4, 0, 3, 3, 7, 4, // Left
            1, 5, 6, 6, 2, 1, // Right
            3, 2, 6, 6, 7, 3, // Top
            4, 5, 1, 1, 0, 4, // Bottom
        ];

        let vertex_buffer = Arc::new(Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            vertices,
        ).unwrap())
        /*.unwrap()*/;

        let index_buffer = Arc::new(
            Buffer::from_iter(
                memory_allocator,
                BufferCreateInfo {
                    usage: BufferUsage::INDEX_BUFFER,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                indices,
            )
            .unwrap(),
        );

        Fighter {
            position,
            rotation: 0.0,
            health: 100.0,
            is_attacking: false,
            vertex_buffer,
            index_buffer,
        }
    }

    fn get_model_matrix(&self) -> Matrix4<f32> {
        let translation = Matrix4::from_translation(self.position);
        let rotation = Matrix4::from_angle_y(Rad(self.rotation));
        translation * rotation
    }

    fn update(&mut self, delta_time: f32) {
        // Update fighter logic (animation, physics, etc.)
        if self.is_attacking {
            // Attack animation logic
        }
    }
}

// Main game state
struct GameState {
    fighter1: Fighter,
    fighter2: Fighter,
    camera_position: Point3<f32>,
    camera_target: Point3<f32>,
}

impl GameState {
    fn new(memory_allocator: Arc<StandardMemoryAllocator>) -> Self {
        GameState {
            fighter1: Fighter::new(memory_allocator.clone(), Vector3::new(-3.0, 0.0, 0.0)),
            fighter2: Fighter::new(memory_allocator.clone(), Vector3::new(3.0, 0.0, 0.0)),
            camera_position: Point3::new(0.0, 2.0, 10.0),
            camera_target: Point3::new(0.0, 0.0, 0.0),
        }
    }

    fn update(&mut self, delta_time: f32) {
        self.fighter1.update(delta_time);
        self.fighter2.update(delta_time);
    }
}

// Vertex shader
mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: r"
            #version 450

            layout(location = 0) in vec3 position;
            layout(location = 1) in vec3 normal;
            layout(location = 2) in vec2 tex_coords;

            layout(push_constant) uniform PushConstants {
                mat4 model;
                mat4 view;
                mat4 projection;
            } pc;

            layout(location = 0) out vec3 frag_normal;
            layout(location = 1) out vec2 frag_tex_coords;

            void main() {
                gl_Position = pc.projection * pc.view * pc.model * vec4(position, 1.0);
                frag_normal = mat3(transpose(inverse(pc.model))) * normal;
                frag_tex_coords = tex_coords;
            }
        "
    }
}

// Fragment shader
mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: r"
            #version 450

            layout(location = 0) in vec3 frag_normal;
            layout(location = 1) in vec2 frag_tex_coords;

            layout(location = 0) out vec4 f_color;

            void main() {
                vec3 light_dir = normalize(vec3(0.5, 1.0, 0.3));
                float diff = max(dot(normalize(frag_normal), light_dir), 0.3);
                vec3 color = vec3(0.8, 0.3, 0.2) * diff;
                f_color = vec4(color, 1.0);
            }
        "
    }
}

fn main() {
    // Initialize Vulkan
    let library = VulkanLibrary::new().expect("Failed to load Vulkan library");
    //let required_extensions = Surface::required_extensions(&library);
    let required_extensions = Surface::required_extensions(&library).unwrap();

    let instance = Instance::new(
        library,
        InstanceCreateInfo {
            enabled_extensions: required_extensions,
            ..Default::default()
        },
    )
    .expect("Failed to create Vulkan instance");

    // Create window
    let event_loop = EventLoop::new();
    let event_loop_window_target = EventLoopWindowTarget::from(event_loop);
    let window = WindowBuilder::new()
        .with_title("3D Fighting Game - Vulkan/Rust")
        .build(&event_loop_window_target)
        .unwrap();

    let arc_window = Arc::new(window);

    let surface = Surface::from_window(instance.clone(), arc_window).unwrap();

    // Select physical device
    let device_extensions = vulkano::device::DeviceExtensions {
        khr_swapchain: true,
        ..Default::default()
    };

    let (physical_device, queue_family_index) = instance
        .enumerate_physical_devices()
        .unwrap()
        .filter(|p| p.supported_extensions().contains(&device_extensions))
        .filter_map(|p| {
            p.queue_family_properties()
                .iter()
                .enumerate()
                .position(|(i, q)| {
                    q.queue_flags.contains(QueueFlags::GRAPHICS)
                        && p.surface_support(i as u32, &surface).unwrap_or(false)
                })
                .map(|i| (p, i as u32))
        })
        .min_by_key(|(p, _)| match p.properties().device_type {
            vulkano::device::physical::PhysicalDeviceType::DiscreteGpu => 0,
            vulkano::device::physical::PhysicalDeviceType::IntegratedGpu => 1,
            _ => 2,
        })
        .expect("No suitable physical device found");

    println!(
        "Using device: {} (type: {:?})",
        physical_device.properties().device_name,
        physical_device.properties().device_type
    );

    // Create logical device and queues
    let (device, mut queues) = Device::new(
        physical_device.clone(),
        DeviceCreateInfo {
            enabled_extensions: device_extensions,
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            ..Default::default()
        },
    )
    .expect("Failed to create device");

    let queue = queues.next().unwrap();

    // Create memory allocator
    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
    // let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
    //     device.clone(),
    //     Default::default(),
    // ));
    let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
        device.clone(),
        Default::default(),
    ));

    // Create swapchain
    let caps = physical_device
        .surface_capabilities(&surface, Default::default())
        .unwrap();
    let dimensions = surface
        .object()
        .unwrap()
        .downcast_ref::<winit::window::Window>()
        .unwrap()
        .inner_size();
    let composite_alpha = caps.supported_composite_alpha.into_iter().next().unwrap();
    let image_format = physical_device
        .surface_formats(&surface, Default::default())
        .unwrap()[0]
        .0;

    let (mut swapchain, images) = Swapchain::new(
        device.clone(),
        surface.clone(),
        SwapchainCreateInfo {
            min_image_count: caps.min_image_count + 1,
            image_format,
            image_extent: dimensions.into(),
            image_usage: ImageUsage::COLOR_ATTACHMENT,
            composite_alpha,
            ..Default::default()
        },
    )
    .unwrap();

    // Create render pass
    let render_pass = vulkano::single_pass_renderpass!(
        device.clone(),
        attachments: {
            color: {
                format: swapchain.image_format(),
                samples: 1,
                load_op: Clear,
                store_op: Store,
            },
        },
        pass: {
            color: [color],
            depth_stencil: {},
        },
    )
    .unwrap();

    // Create graphics pipeline
    let vs = vs::load(device.clone()).unwrap();
    let fs = fs::load(device.clone()).unwrap();

    let pipeline = {
        let vs_entry = vs.entry_point("main").unwrap();
        let fs_entry = fs.entry_point("main").unwrap();

        let vertex_input_state = FighterVertex::per_vertex()
            .definition(&vs_entry.info().input_interface)
            .unwrap();

        let stages = [
            PipelineShaderStageCreateInfo::new(vs_entry),
            PipelineShaderStageCreateInfo::new(fs_entry),
        ];

        let layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                .into_pipeline_layout_create_info(device.clone())
                .unwrap(),
        )
        .unwrap();

        let subpass = Subpass::from(render_pass.clone(), 0).unwrap();

        GraphicsPipeline::new(
            device.clone(),
            None,
            GraphicsPipelineCreateInfo {
                stages: stages.into_iter().collect(),
                vertex_input_state: Some(vertex_input_state),
                input_assembly_state: Some(InputAssemblyState::default()),
                viewport_state: Some(ViewportState::default()),
                rasterization_state: Some(RasterizationState::default()),
                multisample_state: Some(MultisampleState::default()),
                color_blend_state: Some(ColorBlendState::with_attachment_states(
                    subpass.num_color_attachments(),
                    ColorBlendAttachmentState::default(),
                )),
                subpass: Some(subpass.into()),
                ..GraphicsPipelineCreateInfo::layout(layout)
            },
        )
        .unwrap()
    };

    // Create framebuffers
    let mut framebuffers = create_framebuffers(&images, &render_pass);

    // Initialize game state
    let mut game_state = GameState::new(memory_allocator.clone());

    // Main loop
    let mut recreate_swapchain = false;
    let mut previous_frame_end = Some(sync::now(device.clone()).boxed());

    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                *control_flow = ControlFlow::Exit;
            }
            Event::WindowEvent {
                event: WindowEvent::Resized(_),
                ..
            } => {
                recreate_swapchain = true;
            }
            Event::MainEventsCleared => {
                // Update game state
                game_state.update(0.016); // ~60 FPS

                // Render frame
                previous_frame_end.as_mut().unwrap().cleanup_finished();

                if recreate_swapchain {
                    let new_dimensions = surface
                        .object()
                        .unwrap()
                        .downcast_ref::<winit::window::Window>()
                        .unwrap()
                        .inner_size();
                    let (new_swapchain, new_images) = swapchain
                        .recreate(SwapchainCreateInfo {
                            image_extent: new_dimensions.into(),
                            ..swapchain.create_info()
                        })
                        .expect("Failed to recreate swapchain");

                    swapchain = new_swapchain;
                    framebuffers = create_framebuffers(&new_images, &render_pass);
                    recreate_swapchain = false;
                }

                let (image_index, suboptimal, acquire_future) =
                    match vulkano::swapchain::acquire_next_image(swapchain.clone(), None)
                        .map_err(Validated::unwrap)
                    {
                        Ok(r) => r,
                        Err(VulkanError::OutOfDate) => {
                            recreate_swapchain = true;
                            return;
                        }
                        Err(e) => panic!("Failed to acquire next image: {e}"),
                    };

                if suboptimal {
                    recreate_swapchain = true;
                }

                // Build command buffer
                let mut builder = AutoCommandBufferBuilder::primary(
                    //&command_buffer_allocator,
                    command_buffer_allocator,
                    queue.queue_family_index(),
                    CommandBufferUsage::OneTimeSubmit,
                )
                .unwrap();

                let dimensions = images[image_index as usize].extent();

                // Setup camera matrices
                let aspect_ratio = dimensions[0] as f32 / dimensions[1] as f32;
                let proj = perspective(Deg(60.0), aspect_ratio, 0.1, 100.0);
                let view = Matrix4::look_at_rh(
                    game_state.camera_position,
                    game_state.camera_target,
                    Vector3::new(0.0, 1.0, 0.0),
                );

                builder
                    .begin_render_pass(
                        RenderPassBeginInfo {
                            clear_values: vec![Some([0.1, 0.1, 0.15, 1.0].into())],
                            ..RenderPassBeginInfo::framebuffer(
                                framebuffers[image_index as usize].clone(),
                            )
                        },
                        Default::default(),
                    )
                    .unwrap()
                    .set_viewport(
                        0,
                        [Viewport {
                            offset: [0.0, 0.0],
                            extent: [dimensions[0] as f32, dimensions[1] as f32],
                            depth_range: 0.0..=1.0,
                        }]
                        .into_iter()
                        .collect(),
                    )
                    .unwrap()
                    .bind_pipeline_graphics(pipeline.clone())
                    .unwrap();

                // Draw fighter 1
                let push_constants = PushConstants {
                    model: game_state.fighter1.get_model_matrix().into(),
                    view: view.into(),
                    projection: proj.into(),
                };
                unsafe {
                    builder
                        .push_constants(pipeline.layout().clone(), 0, push_constants)
                        .unwrap()
                        .bind_vertex_buffers(0, game_state.fighter1.vertex_buffer.clone())
                        .unwrap()
                        .bind_index_buffer(game_state.fighter1.index_buffer.clone())
                        .unwrap()
                        .draw_indexed(36, 1, 0, 0, 0)
                        .unwrap()
                };

                // Draw fighter 2
                let push_constants = PushConstants {
                    model: game_state.fighter2.get_model_matrix().into(),
                    view: view.into(),
                    projection: proj.into(),
                };
                unsafe {
                    builder
                        .push_constants(pipeline.layout().clone(), 0, push_constants)
                        .unwrap()
                        .bind_vertex_buffers(0, game_state.fighter2.vertex_buffer.clone())
                        .unwrap()
                        .bind_index_buffer(game_state.fighter2.index_buffer.clone())
                        .unwrap()
                        .draw_indexed(36, 1, 0, 0, 0)
                        .unwrap()
                };

                builder.end_render_pass(Default::default()).unwrap();

                let command_buffer = builder.build().unwrap();

                let future = previous_frame_end
                    .take()
                    .unwrap()
                    .join(acquire_future)
                    .then_execute(queue.clone(), command_buffer)
                    .unwrap()
                    .then_swapchain_present(
                        queue.clone(),
                        SwapchainPresentInfo::swapchain_image_index(swapchain.clone(), image_index),
                    )
                    .then_signal_fence_and_flush();

                match future.map_err(Validated::unwrap) {
                    Ok(future) => {
                        previous_frame_end = Some(future.boxed());
                    }
                    Err(VulkanError::OutOfDate) => {
                        recreate_swapchain = true;
                        previous_frame_end = Some(sync::now(device.clone()).boxed());
                    }
                    Err(e) => {
                        println!("Failed to flush future: {e}");
                        previous_frame_end = Some(sync::now(device.clone()).boxed());
                    }
                }
            }
            _ => {}
        }
    });
}

fn create_framebuffers(
    images: &[Arc<Image>],
    render_pass: &Arc<RenderPass>,
) -> Vec<Arc<Framebuffer>> {
    images
        .iter()
        .map(|image| {
            let view = ImageView::new_default(image.clone()).unwrap();
            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![view],
                    ..Default::default()
                },
            )
            .unwrap()
        })
        .collect::<Vec<_>>()
}
